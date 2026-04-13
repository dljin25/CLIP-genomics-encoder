from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "deep_sets"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.deep_sets import DeepSetsEncoder
from src.project_config import load_project_config, resolve_project_path


@dataclass(frozen=True)
class TrainingConfig:
    """Resolved hyperparameters for one training run."""

    name: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    recall_at_k: int
    mask_probability: float
    max_genes: int
    embedding_dim: int
    hidden_dim: int
    dropout: float
    num_heads: int
    num_layers: int
    output_dim: int
    pooling: str | None = "sum"
    pos_weight_max: float = 100.0


@dataclass(frozen=True)
class GeneIndexing:
    """Maps raw gene ids into model input ids with reserved padding and mask ids."""

    raw_vocab_size: int
    pad_index: int = 0

    @property
    def mask_index(self) -> int:
        """Return the reserved mask token id for model inputs."""

        return self.raw_vocab_size + 1

    @property
    def input_vocab_size(self) -> int:
        """Return the embedding vocabulary size including pad and mask ids."""

        return self.raw_vocab_size + 2

    def to_input_ids(self, raw_gene_ids: torch.Tensor) -> torch.Tensor:
        """Shift raw gene ids so zero can be reserved for padding."""

        return raw_gene_ids + 1


def build_training_config(
    project_config: dict[str, Any],
    *,
    model_name: str,
) -> TrainingConfig:
    """Merge shared training settings with model-specific overrides."""

    model_config = project_config["model"]
    training_config = project_config["training"]
    model_specific = model_config.get(model_name, {})
    training_overrides = model_specific.get("training_overrides", {})
    resolved_training = {**training_config, **training_overrides}

    return TrainingConfig(
        name=resolved_training.get("config_name", "default"),
        learning_rate=resolved_training["learning_rate"],
        weight_decay=resolved_training["weight_decay"],
        batch_size=resolved_training["batch_size"],
        epochs=resolved_training["epochs"],
        recall_at_k=resolved_training["recall_at_k"],
        mask_probability=resolved_training["mask_probability"],
        max_genes=resolved_training["max_genes"],
        embedding_dim=resolved_training["embedding_dim"],
        hidden_dim=resolved_training["hidden_dim"],
        dropout=resolved_training["dropout"],
        num_heads=model_specific.get("num_heads", 8),
        num_layers=model_specific.get("num_layers", 2),
        output_dim=model_config["output_dim"],
        pooling=model_specific.get("pooling", "sum") if model_name == "deep_sets" else None,
        pos_weight_max=resolved_training.get("pos_weight_max", 100.0),
    )


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch random number generators."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_metadata_dataframe(
    parquet_path: str | Path,
    *,
    patient_id_column: str,
    gene_index_column: str,
    gene_symbol_column: str | None = None,
) -> pd.DataFrame:
    """Load tokenized metadata rows with nonempty patient ids and gene sets."""

    dataframe = pd.read_parquet(Path(parquet_path))
    required_columns = [patient_id_column, gene_index_column]

    keep_columns = required_columns.copy()
    if gene_symbol_column is not None and gene_symbol_column in dataframe.columns:
        keep_columns.append(gene_symbol_column)

    filtered = dataframe.loc[:, keep_columns].copy()
    filtered[patient_id_column] = filtered[patient_id_column].astype("string").str.strip()
    filtered = filtered[
        filtered[patient_id_column].notna()
        & (filtered[patient_id_column] != "")
        & filtered[gene_index_column].map(lambda value: hasattr(value, "__len__") and len(value) > 0)
    ]
    return filtered.reset_index(drop=True)


def infer_gene_indexing(dataframe: pd.DataFrame, gene_index_column: str) -> GeneIndexing:
    """Infer raw gene vocabulary size from the largest observed gene id."""

    max_gene_index = max(int(gene_id) for gene_ids in dataframe[gene_index_column] for gene_id in gene_ids)
    return GeneIndexing(raw_vocab_size=max_gene_index + 1)


def split_dataframe(
    dataframe: pd.DataFrame,
    *,
    seed: int,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Randomly split patient rows into train, validation, and test sets."""

    indices = list(range(len(dataframe)))
    random.Random(seed).shuffle(indices)
    n_train = int(len(indices) * train_fraction)
    n_val = int(len(indices) * val_fraction)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    return (
        dataframe.iloc[train_indices].reset_index(drop=True),
        dataframe.iloc[val_indices].reset_index(drop=True),
        dataframe.iloc[test_indices].reset_index(drop=True),
    )


class MaskedGeneIndexDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset that masks genes from tokenized patient gene sets."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        gene_index_column: str,
        indexing: GeneIndexing, 
        max_genes: int,
        mask_probability: float,
        random_subsample: bool,
    ) -> None:
        """Store the dataframe and masking settings."""

        self.dataframe = dataframe.reset_index(drop=True)
        self.gene_index_column = gene_index_column
        self.indexing = indexing
        self.max_genes = max_genes
        self.mask_probability = mask_probability
        self.random_subsample = random_subsample

    def __len__(self) -> int:
        """Return the number of patient rows."""

        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one masked gene-set example and its missing-gene target."""

        raw_gene_ids = torch.tensor(
            self.dataframe.iloc[index][self.gene_index_column],
            dtype=torch.long,
        )
        raw_gene_ids = torch.unique(raw_gene_ids, sorted=True)
        if raw_gene_ids.numel() == 0:
            raise ValueError("Encountered an empty gene-index set.")

        if raw_gene_ids.numel() > self.max_genes:
            if self.random_subsample:
                keep_positions = torch.randperm(raw_gene_ids.numel())[: self.max_genes]
                raw_gene_ids = raw_gene_ids[keep_positions]
            else:
                raw_gene_ids = raw_gene_ids[: self.max_genes]

        input_token_ids = self.indexing.to_input_ids(raw_gene_ids)
        num_masked = min(
            max(1, math.ceil(input_token_ids.numel() * self.mask_probability)),
            int(input_token_ids.numel()),
        )
        masked_positions = torch.randperm(input_token_ids.numel())[:num_masked]
        masked_gene_ids = raw_gene_ids[masked_positions]
        input_token_ids = input_token_ids.clone()
        input_token_ids[masked_positions] = self.indexing.mask_index

        target_multihot = torch.zeros(self.indexing.raw_vocab_size, dtype=torch.float32)
        target_multihot[masked_gene_ids] = 1.0

        return {
            "input_token_ids": input_token_ids,
            "attention_mask": torch.ones_like(input_token_ids),
            "target_multihot": target_multihot,
        }


def collate_masked_gene_batch(
    batch: list[dict[str, torch.Tensor]],
    *,
    pad_index: int,
) -> dict[str, torch.Tensor]:
    """Pad variable-length gene sets and stack multi-hot targets."""

    input_token_ids = pad_sequence(
        [item["input_token_ids"] for item in batch],
        batch_first=True,
        padding_value=pad_index,
    )
    attention_mask = pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    targets = torch.stack([item["target_multihot"] for item in batch])
    return {
        "input_token_ids": input_token_ids,
        "attention_mask": attention_mask,
        "target_multihot": targets,
    }


class MaskedGenePredictor(nn.Module):
    """Encoder plus classifier for predicting masked raw gene ids."""

    def __init__(self, encoder: nn.Module, output_dim: int, num_targets: int) -> None:
        """Initialize the encoder wrapper and gene classifier."""

        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(output_dim, num_targets)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(
        self,
        input_token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return gene logits and the patient-set embedding."""

        embedding = self.encoder(input_token_ids, attention_mask)
        logits = self.classifier(embedding)
        return logits, embedding


def build_encoder(
    indexing: GeneIndexing,
    config: TrainingConfig,
) -> nn.Module:
    """Construct the Deep Sets encoder."""

    return DeepSetsEncoder(
        vocab_size=indexing.input_vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        pad_index=indexing.pad_index,
        pooling=config.pooling or "sum",
        dropout=config.dropout,
    )


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    gene_index_column: str,
    indexing: GeneIndexing,
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""

    collate_fn = lambda batch: collate_masked_gene_batch(batch, pad_index=indexing.pad_index)
    train_dataset = MaskedGeneIndexDataset(
        train_df,
        gene_index_column=gene_index_column,
        indexing=indexing,
        max_genes=config.max_genes,
        mask_probability=config.mask_probability,
        random_subsample=True,
    )
    val_dataset = MaskedGeneIndexDataset(
        val_df,
        gene_index_column=gene_index_column,
        indexing=indexing,
        max_genes=config.max_genes,
        mask_probability=config.mask_probability,
        random_subsample=False,
    )
    test_dataset = MaskedGeneIndexDataset(
        test_df,
        gene_index_column=gene_index_column,
        indexing=indexing,
        max_genes=config.max_genes,
        mask_probability=config.mask_probability,
        random_subsample=False,
    )
    return (
        DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn),
        DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn),
        DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn),
    )


def compute_recall_at_k(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    k: int,
) -> float:
    """Compute mean recall for masked genes among the top-k predictions."""

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
    """Compute weighted BCE loss for masked-gene prediction."""

    return nn.functional.binary_cross_entropy_with_logits(
        logits,
        target_multihot,
        pos_weight=pos_weight,
    )


def compute_pos_weight(
    dataframe: pd.DataFrame,
    *,
    gene_index_column: str,
    indexing: GeneIndexing,
    pos_weight_max: float,
) -> torch.Tensor:
    """Estimate capped positive-class weights from training-set gene presence."""

    positive_counts = torch.zeros(indexing.raw_vocab_size, dtype=torch.float32)
    for gene_ids in dataframe[gene_index_column]:
        present_gene_ids = torch.unique(torch.tensor(gene_ids, dtype=torch.long))
        positive_counts[present_gene_ids] += 1.0

    negative_counts = len(dataframe) - positive_counts
    pos_weight = negative_counts / positive_counts.clamp_min(1.0)
    pos_weight = torch.where(positive_counts > 0, pos_weight, torch.ones_like(pos_weight))
    return pos_weight.clamp_max(pos_weight_max)


def summarize_gene_sets(
    dataframe: pd.DataFrame,
    *,
    gene_index_column: str,
    indexing: GeneIndexing,
) -> dict[str, object]:
    """Summarize gene-set sizes and input vocabulary conventions."""

    lengths = dataframe[gene_index_column].map(len)
    return {
        "num_train_rows": len(dataframe),
        "raw_gene_vocab_size": indexing.raw_vocab_size,
        "input_vocab_size": indexing.input_vocab_size,
        "pad_index": indexing.pad_index,
        "mask_index": indexing.mask_index,
        "avg_genes_per_sample": float(lengths.mean()) if len(lengths) else 0.0,
        "median_genes_per_sample": float(lengths.median()) if len(lengths) else 0.0,
        "max_genes_per_sample": int(lengths.max()) if len(lengths) else 0,
    }


def run_epoch(
    model: MaskedGenePredictor,
    dataloader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    recall_at_k: int | None,
    pos_weight: torch.Tensor,
) -> dict[str, float]:
    """Run one training or evaluation epoch and return aggregate metrics."""

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

        if recall_at_k is not None:
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())

    metrics = {"loss": total_loss / max(total_examples, 1)}
    if recall_at_k is not None:
        metrics["recall_at_k"] = compute_recall_at_k(
            torch.cat(all_logits, dim=0),
            torch.cat(all_targets, dim=0),
            k=recall_at_k,
        )

    return metrics


def train_one_configuration(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    gene_index_column: str,
    indexing: GeneIndexing,
    config: TrainingConfig,
    device: torch.device,
    checkpoint_dir: Path,
    checkpoint_filename_template: str,
) -> dict[str, object]:
    """Train one model configuration and evaluate its best validation checkpoint."""

    train_stats = summarize_gene_sets(
        train_df,
        gene_index_column=gene_index_column,
        indexing=indexing,
    )
    pos_weight = compute_pos_weight(
        train_df,
        gene_index_column=gene_index_column,
        indexing=indexing,
        pos_weight_max=config.pos_weight_max,
    ).to(device)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df,
        val_df,
        test_df,
        gene_index_column=gene_index_column,
        indexing=indexing,
        config=config,
    )

    encoder = build_encoder(indexing, config).to(device)
    model = MaskedGenePredictor(
        encoder,
        output_dim=config.output_dim,
        num_targets=indexing.raw_vocab_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_checkpoint_metrics: dict[str, float] | None = None

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            recall_at_k=None,
            pos_weight=pos_weight,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                recall_at_k=None,
                pos_weight=pos_weight,
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
            }
        )

        elapsed_seconds = time.perf_counter() - epoch_start
        print(
            (
                f"Epoch {epoch:03d}/{config.epochs} "
                f"| train_loss={train_metrics['loss']:.4f} "
                f"| val_loss={val_metrics['loss']:.4f} "
                f"| {elapsed_seconds:.1f}s"
            ),
            flush=True,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_checkpoint_metrics = {
                "epoch": float(epoch),
                "val_loss": val_metrics["loss"],
                "train_loss": train_metrics["loss"],
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
            "raw_gene_vocab_size": indexing.raw_vocab_size,
            "input_vocab_size": indexing.input_vocab_size,
            "pad_index": indexing.pad_index,
            "mask_index": indexing.mask_index,
            "gene_input_id_shift": 1,
            "output_dim": config.output_dim,
        },
        checkpoint_path,
    )

    return {
        "config": asdict(config),
        "train_stats": train_stats,
        "history": history,
        "checkpoint_selection_metric": "val_loss",
        "best_checkpoint_epoch": int(best_checkpoint_metrics["epoch"]),
        "best_checkpoint_val_loss": best_checkpoint_metrics["val_loss"],
        "best_checkpoint_train_loss": best_checkpoint_metrics["train_loss"],
        "test_loss": test_metrics["loss"],
        "test_recall_at_k": test_metrics["recall_at_k"],
        "raw_gene_vocab_size": indexing.raw_vocab_size,
        "input_vocab_size": indexing.input_vocab_size,
        "num_targets": indexing.raw_vocab_size,
        "encoder_checkpoint": str(checkpoint_path),
    }


def save_json(data: dict[str, object], output_path: Path) -> None:
    """Write a JSON results file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))


def main() -> None:
    """Parse CLI arguments and run training."""

    parser = argparse.ArgumentParser(
        description="Train a genomics encoder from tokenized metadata parquet."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Project config file. Defaults to config.yaml in the repo root.",
    )
    preliminary_args, _ = parser.parse_known_args()
    project_config = load_project_config(preliminary_args.config)
    data_config = project_config["data"]
    schema_config = data_config["schema"]
    output_config = project_config["outputs"]
    runtime_config = project_config["runtime"]
    split_config = project_config["splits"]

    default_input_parquet = resolve_project_path(data_config["raw"]["metadata_parquet"])
    default_output_dir = resolve_project_path(output_config["runs_dir"])
    default_checkpoint_dir = resolve_project_path(output_config["checkpoints_dir"])

    parser.add_argument(
        "input_parquet",
        nargs="?",
        type=Path,
        default=default_input_parquet,
        help="Metadata parquet containing gene_symbol_index_list.",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--checkpoint-dir", type=Path, default=default_checkpoint_dir)
    parser.add_argument("--seed", type=int, default=runtime_config["seed"])
    args = parser.parse_args()

    set_seed(args.seed)
    filtered_df = load_metadata_dataframe(
        args.input_parquet,
        patient_id_column=schema_config["patient_id_column"],
        gene_index_column=schema_config["gene_index_column"],
        gene_symbol_column=schema_config.get("gene_symbol_column"),
    )
    train_df, val_df, test_df = split_dataframe(
        filtered_df,
        seed=args.seed,
        train_fraction=split_config["train_fraction"],
        val_fraction=split_config["val_fraction"],
    )

    indexing = infer_gene_indexing(filtered_df, schema_config["gene_index_column"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_config = build_training_config(project_config, model_name=MODEL_NAME)

    runs: list[dict[str, object]] = []
    results: dict[str, object] = {
        "model": MODEL_NAME,
        "config_path": str(args.config),
        "input_parquet": str(args.input_parquet),
        "gene_index_column": schema_config["gene_index_column"],
        "seed": args.seed,
        "device": str(device),
        "num_rows": len(filtered_df),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "split_fractions": {
            "train": split_config["train_fraction"],
            "val": split_config["val_fraction"],
            "test": split_config["test_fraction"],
        },
        "raw_gene_vocab_size": indexing.raw_vocab_size,
        "input_vocab_size": indexing.input_vocab_size,
        "pad_index": indexing.pad_index,
        "mask_index": indexing.mask_index,
        "gene_input_id_shift": 1,
        "runs": runs,
    }

    runs.append(
        train_one_configuration(
            MODEL_NAME,
            train_df,
            val_df,
            test_df,
            gene_index_column=schema_config["gene_index_column"],
            indexing=indexing,
            config=training_config,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_filename_template=output_config["checkpoint_filename_template"],
        )
    )

    output_path = args.output_dir / output_config["results_filename_template"].format(
        model=MODEL_NAME,
        config_name=training_config.name,
    )
    save_json(results, output_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
