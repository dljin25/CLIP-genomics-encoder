from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.build_gene_vocabulary import GeneVocabulary
from src.data.tokenize_gene_dataset import TokenizedGeneSetDataset
from src.models.deep_sets import DeepSetsEncoder
from src.models.set_transformer import SetTransformerEncoder
from src.project_config import load_project_config, resolve_project_path

PROJECT = load_project_config()
SCHEMA = PROJECT["data"]["schema"]
PATIENT_ID = SCHEMA["patient_id_column"]
VARIANT_SEQUENCE = SCHEMA["variant_sequence_column"]
DATASET = SCHEMA["dataset_column"]
CANCER_TYPE = SCHEMA["cancer_type_column"]
DEFAULT_MIN_FREQUENCY = PROJECT["training"]["min_frequency"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TrainingConfig:
    name: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    min_frequency: int = DEFAULT_MIN_FREQUENCY
    recall_at_k: int = 10
    mask_probability: float = 0.15
    max_genes: int = 256
    embedding_dim: int = 256
    hidden_dim: int = 1024
    dropout: float = 0.1
    num_heads: int = 8
    num_layers: int = 2
    output_dim: int = 512
    pooling: str | None = "mean"


def build_training_config(
    project_config: dict[str, Any],
    *,
    model_name: str,
) -> TrainingConfig:
    model_config = project_config["model"]
    training_config = project_config["training"]
    model_specific = model_config.get(model_name, {})
    training_overrides = model_specific.get("training_overrides", {})

    resolved_training = {
        **training_config,
        **training_overrides,
    }
    return TrainingConfig(
        name=resolved_training.get("config_name", "default"),
        learning_rate=resolved_training["learning_rate"],
        weight_decay=resolved_training["weight_decay"],
        batch_size=resolved_training["batch_size"],
        epochs=resolved_training["epochs"],
        min_frequency=resolved_training["min_frequency"],
        recall_at_k=resolved_training["recall_at_k"],
        mask_probability=resolved_training["mask_probability"],
        max_genes=resolved_training["max_genes"],
        embedding_dim=resolved_training["embedding_dim"],
        hidden_dim=resolved_training["hidden_dim"],
        dropout=resolved_training["dropout"],
        num_heads=model_specific.get("num_heads", 8),
        num_layers=model_specific.get("num_layers", 2),
        output_dim=model_config["output_dim"],
        pooling=model_specific.get("pooling") if model_name == "deep_sets" else None,
    )


class MaskedGeneSetDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        vocabulary: GeneVocabulary,
        *,
        max_genes: int,
        mask_probability: float,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.vocabulary = vocabulary
        self.max_genes = max_genes
        self.mask_probability = mask_probability
        self.target_gene_ids = vocabulary.gene_token_ids()
        self.target_index_by_token_id = {
            token_id: idx for idx, token_id in enumerate(self.target_gene_ids)
        }

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        genes = TokenizedGeneSetDataset.parse_variant_sequence(row[VARIANT_SEQUENCE])
        token_ids = self.vocabulary.encode(genes, self.max_genes)
        input_token_ids = token_ids.clone()
        attention_mask = (token_ids != self.vocabulary.pad_index).long()

        non_pad_positions = torch.nonzero(token_ids != self.vocabulary.pad_index, as_tuple=False).flatten()
        if non_pad_positions.numel() == 0:
            raise ValueError("Encountered an empty mutation set after filtering.")

        known_gene_positions = torch.nonzero(
            (token_ids != self.vocabulary.pad_index)
            & (token_ids != self.vocabulary.unk_index)
            & (token_ids != self.vocabulary.mask_index),
            as_tuple=False,
        ).flatten()
        mask_source_positions = known_gene_positions if known_gene_positions.numel() > 0 else non_pad_positions

        num_masked = min(
            max(1, math.ceil(mask_source_positions.numel() * self.mask_probability)),
            int(mask_source_positions.numel()),
        )
        permutation = torch.randperm(mask_source_positions.numel())
        masked_positions = mask_source_positions[permutation[:num_masked]]
        masked_token_ids = token_ids[masked_positions]
        input_token_ids[masked_positions] = self.vocabulary.mask_index

        target_multihot = torch.zeros(len(self.target_gene_ids), dtype=torch.float32)
        for token_id in masked_token_ids.tolist():
            target_index = self.target_index_by_token_id.get(token_id)
            if target_index is not None:
                target_multihot[target_index] = 1.0

        return {
            "input_token_ids": input_token_ids,
            "attention_mask": attention_mask,
            "target_multihot": target_multihot,
        }


class MaskedGenePredictor(nn.Module):
    def __init__(self, encoder: nn.Module, output_dim: int, vocabulary: GeneVocabulary) -> None:
        super().__init__()
        self.encoder = encoder
        self.target_gene_ids = vocabulary.gene_token_ids()
        self.classifier = nn.Linear(output_dim, len(self.target_gene_ids))
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(
        self,
        input_token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(input_token_ids, attention_mask)
        logits = self.classifier(embedding)
        return logits, embedding


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_patient_split_key(row: pd.Series) -> str:
    return f"{row[DATASET]}:{row[PATIENT_ID]}"


def compute_split_counts(
    total_count: int,
    *,
    train_fraction: float,
    val_fraction: float,
) -> tuple[int, int, int]:

    n_train = int(total_count * train_fraction)
    n_val = int(total_count * val_fraction)
    n_test = total_count - n_train - n_val
    return n_train, n_val, n_test


def split_dataframe(
    dataframe: pd.DataFrame,
    *,
    seed: int,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    unknown_cancer_type: str = "unknown",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keyed_dataframe = dataframe.copy()
    keyed_dataframe["_patient_split_key"] = keyed_dataframe.apply(build_patient_split_key, axis=1)
    keyed_dataframe["_stratify_label"] = keyed_dataframe[CANCER_TYPE].fillna(unknown_cancer_type).astype("string").str.strip()

    train_ids: set[str] = set()
    val_ids: set[str] = set()
    test_ids: set[str] = set()
    rng = random.Random(seed)

    for _, group in keyed_dataframe.groupby("_stratify_label", sort=True):
        patient_keys = list(pd.unique(group["_patient_split_key"]))
        rng.shuffle(patient_keys)
        n_train, n_val, n_test = compute_split_counts(
            len(patient_keys),
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )
        train_ids.update(patient_keys[:n_train])
        val_ids.update(patient_keys[n_train:n_train + n_val])
        test_ids.update(patient_keys[n_train + n_val:n_train + n_val + n_test])

    return (
        keyed_dataframe[keyed_dataframe["_patient_split_key"].isin(train_ids)].drop(columns=["_patient_split_key", "_stratify_label"]).reset_index(drop=True),
        keyed_dataframe[keyed_dataframe["_patient_split_key"].isin(val_ids)].drop(columns=["_patient_split_key", "_stratify_label"]).reset_index(drop=True),
        keyed_dataframe[keyed_dataframe["_patient_split_key"].isin(test_ids)].drop(columns=["_patient_split_key", "_stratify_label"]).reset_index(drop=True),
    )


def summarize_counts(dataframe: pd.DataFrame, column_name: str) -> dict[str, int]:
    return {
        str(name): int(count)
        for name, count in dataframe[column_name].value_counts().sort_index().items()
    }


def compute_train_stats(
    dataframe: pd.DataFrame,
    vocabulary: GeneVocabulary,
) -> dict[str, object]:
    gene_lists = [TokenizedGeneSetDataset.parse_variant_sequence(value) for value in dataframe[VARIANT_SEQUENCE]]
    gene_counts = [len(genes) for genes in gene_lists]
    flattened = [gene.lower() for genes in gene_lists for gene in genes]
    top_genes = pd.Series(flattened).value_counts().head(10).to_dict() if flattened else {}
    return {
        "num_train_rows": len(dataframe),
        "num_unique_train_patients": int(dataframe.apply(build_patient_split_key, axis=1).nunique()),
        "train_rows_by_dataset": summarize_counts(dataframe, DATASET),
        "vocab_size": len(vocabulary),
        "avg_genes_per_sample": float(sum(gene_counts) / len(gene_counts)) if gene_counts else 0.0,
        "median_genes_per_sample": float(pd.Series(gene_counts).median()) if gene_counts else 0.0,
        "max_genes_per_sample": max(gene_counts) if gene_counts else 0,
        "top_10_genes": top_genes,
    }


def build_encoder(
    model_name: str,
    vocabulary: GeneVocabulary,
    config: TrainingConfig,
) -> nn.Module:
    if model_name == "deep_sets":
        return DeepSetsEncoder(
            vocab_size=len(vocabulary),
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            pad_index=vocabulary.pad_index,
            dropout=config.dropout,
        )
    return SetTransformerEncoder(
        vocab_size=len(vocabulary),
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        pad_index=vocabulary.pad_index,
        dropout=config.dropout,
    )


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    vocabulary: GeneVocabulary,
    *,
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = MaskedGeneSetDataset(
        train_df,
        vocabulary,
        max_genes=config.max_genes,
        mask_probability=config.mask_probability,
    )
    val_dataset = MaskedGeneSetDataset(
        val_df,
        vocabulary,
        max_genes=config.max_genes,
        mask_probability=config.mask_probability,
    )
    test_dataset = MaskedGeneSetDataset(
        test_df,
        vocabulary,
        max_genes=config.max_genes,
        mask_probability=config.mask_probability,
    )
    return (
        DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False),
    )


def compute_recall_at_k(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    k: int,
) -> float:
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape.")
    if k <= 0:
        raise ValueError("k must be positive.")

    recalls: list[float] = []
    top_k = min(k, logits.size(1))
    top_k_indices = torch.topk(logits, k=top_k, dim=1).indices

    for sample_idx in range(targets.size(0)):
        positive_indices = torch.nonzero(targets[sample_idx] > 0, as_tuple=False).flatten()
        if positive_indices.numel() == 0:
            continue
        retrieved = set(top_k_indices[sample_idx].tolist())
        hits = sum(1 for index in positive_indices.tolist() if index in retrieved)
        recalls.append(hits / positive_indices.numel())

    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def compute_masked_gene_loss(
    logits: torch.Tensor,
    target_multihot: torch.Tensor,
    *,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(
        logits,
        target_multihot,
        pos_weight=pos_weight,
    )


def compute_pos_weight(
    dataframe: pd.DataFrame,
    vocabulary: GeneVocabulary,
) -> torch.Tensor:
    target_gene_ids = vocabulary.gene_token_ids()
    target_index_by_token_id = {
        token_id: idx for idx, token_id in enumerate(target_gene_ids)
    }
    positive_counts = torch.zeros(len(target_gene_ids), dtype=torch.float32)

    for value in dataframe[VARIANT_SEQUENCE]:
        genes = TokenizedGeneSetDataset.parse_variant_sequence(value)
        present_token_ids = {
            vocabulary.token_to_id[gene.upper()]
            for gene in genes
            if gene.upper() in vocabulary.token_to_id
        }
        for token_id in present_token_ids:
            target_index = target_index_by_token_id.get(token_id)
            if target_index is not None:
                positive_counts[target_index] += 1.0

    negative_counts = len(dataframe) - positive_counts
    pos_weight = negative_counts / positive_counts.clamp_min(1.0)
    return torch.where(positive_counts > 0, pos_weight, torch.ones_like(pos_weight))


def run_epoch(
    model: MaskedGenePredictor,
    dataloader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    recall_at_k: int,
    pos_weight: torch.Tensor,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_examples = 0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for batch in dataloader:
        input_token_ids = batch["input_token_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target_multihot"].to(device)

        logits, _ = model(input_token_ids, attention_mask)
        loss = compute_masked_gene_loss(logits, targets, pos_weight=pos_weight)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = input_token_ids.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

    return {
        "loss": total_loss / max(total_examples, 1),
        "recall_at_k": compute_recall_at_k(
            torch.cat(all_logits, dim=0),
            torch.cat(all_targets, dim=0),
            k=recall_at_k,
        ),
    }


def train_one_configuration(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    config: TrainingConfig,
    device: torch.device,
    checkpoint_dir: Path,
    checkpoint_filename_template: str,
) -> dict[str, object]:
    vocabulary = GeneVocabulary.build(
        (TokenizedGeneSetDataset.parse_variant_sequence(value) for value in train_df[VARIANT_SEQUENCE]),
        min_frequency=config.min_frequency,
    )
    target_gene_ids = vocabulary.gene_token_ids()
    train_stats = compute_train_stats(train_df, vocabulary)
    pos_weight = compute_pos_weight(train_df, vocabulary).to(device)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df,
        val_df,
        test_df,
        vocabulary,
        config=config,
    )

    encoder = build_encoder(model_name, vocabulary, config).to(device)
    model = MaskedGenePredictor(encoder, output_dim=config.output_dim, vocabulary=vocabulary).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, float]] = []
    best_val_recall_at_k = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_checkpoint_metrics: dict[str, float] | None = None

    for epoch in range(1, config.epochs + 1):
        _ = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            recall_at_k=config.recall_at_k,
            pos_weight=pos_weight,
        )
        with torch.no_grad():
            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer=None,
                device=device,
                recall_at_k=config.recall_at_k,
                pos_weight=pos_weight,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                recall_at_k=config.recall_at_k,
                pos_weight=pos_weight,
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_recall_at_k": train_metrics["recall_at_k"],
                "val_loss": val_metrics["loss"],
                "val_recall_at_k": val_metrics["recall_at_k"],
            }
        )

        if val_metrics["recall_at_k"] > best_val_recall_at_k:
            best_val_recall_at_k = val_metrics["recall_at_k"]
            best_checkpoint_metrics = {
                "epoch": float(epoch),
                "val_loss": val_metrics["loss"],
                "val_recall_at_k": val_metrics["recall_at_k"],
                "train_loss": train_metrics["loss"],
                "train_recall_at_k": train_metrics["recall_at_k"],
            }
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None or best_checkpoint_metrics is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    with torch.no_grad():
        test_metrics = run_epoch(
            model,
            test_loader,
            optimizer=None,
            device=device,
            recall_at_k=config.recall_at_k,
            pos_weight=pos_weight,
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / checkpoint_filename_template.format(
        model=model_name,
        config_name=config.name,
    )
    torch.save(
        {
            "model_name": model_name,
            "config": asdict(config),
            "encoder_state_dict": model.encoder.state_dict(),
            "vocabulary_token_to_id": vocabulary.token_to_id,
            "pad_index": vocabulary.pad_index,
            "unk_index": vocabulary.unk_index,
            "mask_index": vocabulary.mask_index,
            "output_dim": config.output_dim,
        },
        checkpoint_path,
    )

    return {
        "config": asdict(config),
        "train_stats": train_stats,
        "history": history,
        "checkpoint_selection_metric": "val_recall_at_k",
        "best_checkpoint_epoch": int(best_checkpoint_metrics["epoch"]),
        "best_checkpoint_val_loss": best_checkpoint_metrics["val_loss"],
        "best_checkpoint_val_recall_at_k": best_checkpoint_metrics["val_recall_at_k"],
        "best_checkpoint_train_loss": best_checkpoint_metrics["train_loss"],
        "best_checkpoint_train_recall_at_k": best_checkpoint_metrics["train_recall_at_k"],
        "test_loss": test_metrics["loss"],
        "test_recall_at_k": test_metrics["recall_at_k"],
        "vocabulary_size": len(vocabulary),
        "num_targets": len(target_gene_ids),
        "encoder_checkpoint": str(checkpoint_path),
    }


def save_json(data: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))


def load_training_config(config_path: str | Path) -> tuple[dict[str, Any], TrainingConfig]:
    project_config = load_project_config(config_path)
    split_config = project_config["splits"]
    model_choice = project_config["model"]["choice"]
    runtime_config = build_training_config(project_config, model_name=model_choice)
    project_config["_resolved_model_choice"] = model_choice
    project_config["_resolved_split_config"] = split_config
    return project_config, runtime_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a genomics encoder from a processed pooled parquet."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Project config file. Defaults to config.yaml in the repo root.",
    )
    preliminary_args, _ = parser.parse_known_args()
    project_config, default_training_config = load_training_config(preliminary_args.config)
    data_config = project_config["data"]
    output_config = project_config["outputs"]
    runtime_config = project_config["runtime"]
    split_config = project_config["_resolved_split_config"]
    label_config = data_config["labels"]

    default_input_parquet = resolve_project_path(data_config["processed"]["pooled_parquet"])
    default_output_dir = resolve_project_path(output_config["runs_dir"])
    default_checkpoint_dir = resolve_project_path(output_config["checkpoints_dir"])

    parser = argparse.ArgumentParser(
        description="Train a genomics encoder from a processed pooled parquet."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Project config file. Defaults to config.yaml in the repo root.",
    )
    parser.add_argument(
        "input_parquet",
        nargs="?",
        type=Path,
        default=default_input_parquet,
        help="Processed pooled parquet created by src/data/build_parquet.py.",
    )
    parser.add_argument(
        "--model",
        choices=("deep_sets", "set_transformer"),
        default=project_config["_resolved_model_choice"],
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--checkpoint-dir", type=Path, default=default_checkpoint_dir)
    parser.add_argument("--seed", type=int, default=runtime_config["seed"])
    args = parser.parse_args()

    set_seed(args.seed)
    filtered_df = TokenizedGeneSetDataset.load_training_dataframe(args.input_parquet)
    train_df, val_df, test_df = split_dataframe(
        filtered_df,
        seed=args.seed,
        train_fraction=split_config["train_fraction"],
        val_fraction=split_config["val_fraction"],
        unknown_cancer_type=label_config["unknown_cancer_type"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs: list[dict[str, object]] = []
    results: dict[str, object] = {
        "model": args.model,
        "config_path": str(args.config),
        "input_parquet": str(args.input_parquet),
        "seed": args.seed,
        "device": str(device),
        "num_rows": len(filtered_df),
        "rows_by_dataset": summarize_counts(filtered_df, DATASET),
        "rows_by_cancer_type": summarize_counts(filtered_df, CANCER_TYPE),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "runs": runs,
    }

    training_config = build_training_config(project_config, model_name=args.model)

    for config in (training_config,):
        runs.append(
            train_one_configuration(
                args.model,
                train_df,
                val_df,
                test_df,
                config=config,
                device=device,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_filename_template=output_config["checkpoint_filename_template"],
            )
        )

    output_path = args.output_dir / output_config["results_filename_template"].format(
        model=args.model,
        config_name=training_config.name,
    )
    save_json(results, output_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
