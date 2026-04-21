from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from ruamel.yaml import YAML
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIM = 512

from src.models.deep_sets import DeepSetsEncoder
from src.training.gene_dataset import (
    GeneIndexing,
    MaskedGeneIndexDataset,
    collate_masked_gene_batch,
)
from src.training.training_config import TrainingConfig, build_training_config


class MaskedGenePredictor(nn.Module):
    def __init__(self, encoder: nn.Module, num_targets: int) -> None:
        """Initialize the encoder-backed masked gene classifier."""
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(OUTPUT_DIM, num_targets)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, input_token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return prediction logits and encoder embeddings for token IDs."""
        embedding = self.encoder(input_token_ids)
        return self.classifier(embedding), embedding


def split_dataframe(
    dataframe: pd.DataFrame,
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shuffle and split a dataframe into train, validation, and test sets."""
    indices = list(range(len(dataframe)))
    random.Random(seed).shuffle(indices)
    n_train = int(len(indices) * train_fraction)
    n_val = int(len(indices) * val_fraction)
    return (
        dataframe.iloc[indices[:n_train]].reset_index(drop=True),
        dataframe.iloc[indices[n_train:n_train + n_val]].reset_index(drop=True),
        dataframe.iloc[indices[n_train + n_val:]].reset_index(drop=True),
    )


def compute_recall_at_ks(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ks: tuple[int, ...],
) -> dict[int, float]:
    """Compute mean recall for each requested top-k cutoff."""
    top_k = min(max(ks), logits.size(1))
    top_k_indices = torch.topk(logits, k=top_k, dim=1).indices
    recalls: dict[int, list[float]] = {k: [] for k in sorted(set(ks))}

    for sample_idx in range(targets.size(0)):
        positives = torch.nonzero(targets[sample_idx] > 0, as_tuple=False).flatten().tolist()
        for k in recalls:
            retrieved = set(top_k_indices[sample_idx, : min(k, top_k)].tolist())
            recalls[k].append(sum(gene_id in retrieved for gene_id in positives) / len(positives))

    return {k: float(sum(values) / len(values)) for k, values in recalls.items()}


def compute_pos_weight(
    dataframe: pd.DataFrame,
    gene_index_column: str,
    indexing: GeneIndexing,
    pos_weight_max: float,
) -> torch.Tensor:
    """Estimate capped positive class weights from gene frequencies."""
    positive_counts = torch.zeros(indexing.raw_vocab_size, dtype=torch.float32)
    for gene_ids in dataframe[gene_index_column]:
        positive_counts[torch.unique(torch.tensor(gene_ids, dtype=torch.long))] += 1.0
    pos_weight = (len(dataframe) - positive_counts) / positive_counts.clamp_min(1.0)
    return torch.where(positive_counts > 0, pos_weight, torch.ones_like(pos_weight)).clamp_max(pos_weight_max)


def run_epoch(
    model: MaskedGenePredictor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    pos_weight: torch.Tensor,
    recall_at_ks: tuple[int, ...] = (),
) -> dict[str, float]:
    """Run one training or evaluation epoch and return aggregate metrics."""

    model.train(optimizer is not None)
    total_loss = 0.0
    total_examples = 0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for batch in dataloader:
        input_token_ids = batch["input_token_ids"].to(device)
        targets = batch["target_multihot"].to(device)

        logits, _ = model(input_token_ids) # [B, N]
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

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

    metrics = {"loss": total_loss / total_examples}
    if recall_at_ks:
        metrics.update({
            f"recall_at_{k}": value
            for k, value in compute_recall_at_ks(
                torch.cat(all_logits),
                torch.cat(all_targets),
                recall_at_ks,
            ).items()
        })
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
) -> dict[str, object]:
    """Train a masked gene predictor and summarize the best run."""
    collate_fn = lambda batch: collate_masked_gene_batch(batch, pad_index=indexing.pad_index)
    train_loader, val_loader, test_loader = (
        DataLoader(
            MaskedGeneIndexDataset(
                dataframe,
                gene_index_column=gene_index_column,
                indexing=indexing,
                max_genes=config.max_genes,
                mask_probability=config.mask_probability,
                random_subsample=random_subsample,
            ),
            batch_size=config.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )
        for dataframe, shuffle, random_subsample in (
            (train_df, True, True),
            (val_df, False, False),
            (test_df, False, False),
        )
    )

    model = MaskedGenePredictor(
        DeepSetsEncoder(
            vocab_size=indexing.input_vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            output_dim=OUTPUT_DIM,
            pad_index=indexing.pad_index,
            dropout=config.dropout,
        ),
        num_targets=indexing.raw_vocab_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    pos_weight = compute_pos_weight(train_df, gene_index_column, indexing, config.pos_weight_max).to(device)

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_train_loss = 0.0
    best_state = model.state_dict()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, optimizer, device, pos_weight)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, device, pos_weight)
        history.append({"epoch": epoch, "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"]})
        
        print(
            f"Epoch {epoch:03d}/{config.epochs} "
            f"| train_loss={train_metrics['loss']:.4f} "
            f"| val_loss={val_metrics['loss']:.4f} "
            f"| {time.perf_counter() - epoch_start:.1f}s",
            flush=True,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_train_loss = train_metrics["loss"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    with torch.no_grad():
        test_metrics = run_epoch(model, test_loader, None, device, pos_weight, config.recall_at_ks)

    lengths = train_df[gene_index_column].map(len)
    return {
        "config": asdict(config),
        "train_stats": {
            "num_train_rows": len(train_df),
            "raw_gene_vocab_size": indexing.raw_vocab_size,
            "input_vocab_size": indexing.input_vocab_size,
            "pad_index": indexing.pad_index,
            "mask_index": indexing.mask_index,
            "avg_genes_per_sample": float(lengths.mean()),
            "median_genes_per_sample": float(lengths.median()),
            "max_genes_per_sample": int(lengths.max()),
        },
        "history": history,
        "best_model_selection_metric": "val_loss",
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_train_loss": best_train_loss,
        "test_loss": test_metrics["loss"],
        "raw_gene_vocab_size": indexing.raw_vocab_size,
        "input_vocab_size": indexing.input_vocab_size,
        "num_targets": indexing.raw_vocab_size,
        **{f"test_recall_at_{k}": test_metrics[f"recall_at_{k}"] for k in config.recall_at_ks},
    }


def main() -> None:
    """Parse configuration, train the encoder, and write run results."""
    preliminary_parser = argparse.ArgumentParser(add_help=False)
    preliminary_parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    args, _ = preliminary_parser.parse_known_args()

    with args.config.open("r", encoding="utf-8") as file:
        project_config = YAML(typ="safe").load(file)

    data_config = project_config["data"]
    output_config = project_config["outputs"]
    runtime_config = project_config["runtime"]
    parser = argparse.ArgumentParser(description="Train a genomics encoder from tokenized metadata parquet.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("input_parquet", nargs="?", type=Path, default=Path(data_config["raw"]))
    parser.add_argument("--output-dir", type=Path, default=Path(output_config["runs_dir"]))
    parser.add_argument("--seed", type=int, default=runtime_config["seed"])
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    df = pd.read_parquet(args.input_parquet) # read parquet file
    schema_config = project_config["data"]["schema"]
    split_config = project_config["splits"]
    train_df, val_df, test_df = split_dataframe(
        df,
        seed=args.seed,
        train_fraction=split_config["train"],
        val_fraction=split_config["val"],
    )

    gene_index_column = schema_config["gene_index_column"]
    indexing = GeneIndexing(
        raw_vocab_size=max(int(gene_id) for gene_ids in df[gene_index_column] for gene_id in gene_ids) + 1
    )
    config = build_training_config(project_config)
    model_name = project_config["training"]["model_name"]
    runs = [
        train(
            model_name,
            train_df,
            val_df,
            test_df,
            gene_index_column=gene_index_column,
            indexing=indexing,
            config=config,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
    ]
    results = {
        "model": model_name,
        "config_path": str(args.config),
        "input_parquet": str(args.input_parquet),
        "gene_index_column": gene_index_column,
        "seed": args.seed,
        "num_rows": len(df),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "split_fractions": split_config,
        "raw_gene_vocab_size": indexing.raw_vocab_size,
        "input_vocab_size": indexing.input_vocab_size,
        "pad_index": indexing.pad_index,
        "mask_index": indexing.mask_index,
        "runs": runs,
    }

    output_path = args.output_dir / output_config["results_filename_template"].format(
        model=model_name,
        config_name=config.name,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
