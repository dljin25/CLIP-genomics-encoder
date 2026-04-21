from __future__ import annotations 

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.models.deep_sets import DeepSetsEncoder
from src.project_config import load_project_config, resolve_project_path
from src.training.gene_dataset import (
    GeneIndexing,
    MaskedGeneIndexDataset,
    collate_masked_gene_batch,
)
from src.training.training_config import TrainingConfig, build_training_config


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch random number generators."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_dataframe(
    dataframe: pd.DataFrame,
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return gene logits and the patient-set embedding."""

        embedding = self.encoder(input_token_ids)
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


def compute_recall_at_ks(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ks: tuple[int, ...],
) -> dict[int, float]:
    """Compute mean recall for masked genes among several top-k cutoffs."""

    if not ks:
        return {}

    sorted_ks = tuple(sorted(set(ks)))
    top_k = min(max(sorted_ks), logits.size(1))
    top_k_indices = torch.topk(logits, k=top_k, dim=1).indices
    recalls_by_k: dict[int, list[float]] = {k: [] for k in sorted_ks}

    for sample_idx in range(targets.size(0)):
        positive_indices = torch.nonzero(targets[sample_idx] > 0, as_tuple=False).flatten()
        if positive_indices.numel() == 0:
            continue
        positive_ids = positive_indices.tolist()
        for k in sorted_ks:
            retrieved = set(top_k_indices[sample_idx, : min(k, top_k)].tolist())
            hits = sum(1 for index in positive_ids if index in retrieved)
            recalls_by_k[k].append(hits / positive_indices.numel())

    return {
        k: float(sum(recalls) / len(recalls)) if recalls else 0.0
        for k, recalls in recalls_by_k.items()
    }


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
    recall_at_ks: tuple[int, ...],
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
        targets = batch["target_multihot"].to(device)

        logits, _ = model(input_token_ids)
        loss = nn.functional.binary_cross_entropy_with_logits( # BCEWithLogitsLoss
            input=logits, 
            target=targets, 
            pos_weight=pos_weight
        ) 

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = input_token_ids.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        if recall_at_ks:
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())

    metrics = {"loss": total_loss / max(total_examples, 1)}
    if recall_at_ks:
        recall_metrics = compute_recall_at_ks(
            torch.cat(all_logits, dim=0),
            torch.cat(all_targets, dim=0),
            ks=recall_at_ks,
        )
        metrics.update({f"recall_at_{k}": value for k, value in recall_metrics.items()})

    return metrics


def train(
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
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            recall_at_ks=(),
            pos_weight=pos_weight,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                recall_at_ks=(),
                pos_weight=pos_weight,
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
            }
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
            recall_at_ks=config.recall_at_ks,
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
        "raw_gene_vocab_size": indexing.raw_vocab_size,
        "input_vocab_size": indexing.input_vocab_size,
        "num_targets": indexing.raw_vocab_size,
        "encoder_checkpoint": str(checkpoint_path),
        **{
            f"test_recall_at_{k}": test_metrics[f"recall_at_{k}"]
            for k in config.recall_at_ks
        },
    }


def save_json(data: dict[str, object], output_path: Path) -> None:
    """Write a JSON results file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))


def main() -> None:
    """Parse CLI arguments and run training."""

    preliminary_parser = argparse.ArgumentParser(add_help=False)
    preliminary_parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
    )
    preliminary_args, _ = preliminary_parser.parse_known_args()

    project_config = load_project_config(preliminary_args.config)
    data_config = project_config["data"]
    schema_config = data_config["schema"]
    output_config = project_config["outputs"]
    runtime_config = project_config["runtime"]
    split_config = project_config["splits"]

    parser = argparse.ArgumentParser(
        description="Train a genomics encoder from tokenized metadata parquet."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Project config file. Defaults to config.yaml in the repo root.",
    )

    default_input_parquet = resolve_project_path(data_config["raw"])
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
    df = pd.read_parquet(args.input_parquet)

    train_df, val_df, test_df = split_dataframe(
        df,
        seed=args.seed,
        train_fraction=split_config["train"],
        val_fraction=split_config["val"],
    )

    max_index = max(int(gene_id) for gene_ids in df[schema_config["gene_index_column"]] for gene_id in gene_ids)
    indexing = GeneIndexing(raw_vocab_size=max_index + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_config = build_training_config(project_config)
    model_name = project_config["training"]["model_name"]

    runs: list[dict[str, object]] = []
    results: dict[str, object] = {
        "model": model_name,
        "config_path": str(args.config),
        "input_parquet": str(args.input_parquet),
        "gene_index_column": schema_config["gene_index_column"],
        "seed": args.seed,
        "num_rows": len(df),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "split_fractions": {
            "train": split_config["train"],
            "val": split_config["val"],
            "test": split_config["test"],
        },
        "raw_gene_vocab_size": indexing.raw_vocab_size,
        "input_vocab_size": indexing.input_vocab_size,
        "pad_index": indexing.pad_index,
        "mask_index": indexing.mask_index,
        "runs": runs,
    }

    runs.append(
        train(
            model_name,
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
        model=model_name,
        config_name=training_config.name,
    )
    save_json(results, output_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
