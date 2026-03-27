from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from gene_vocabulary import GeneVocabulary, GenomicsParquetDataset
from genomics_deep_sets_encoder import DeepSetsEncoder
from genomics_set_transformer_encoder import SetTransformerEncoder

OUTPUT_DIM = 512

@dataclass(frozen=True)
class TrainingConfig:
    name: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    min_frequency: int = 3              # tweak?
    recall_at_k: int = 10
    mask_probability: float = 0.15      # tweak?
    max_genes: int = 256
    embedding_dim: int = 256
    hidden_dim: int = 1024
    dropout: float = 0.1
    num_heads: int = 8
    num_layers: int = 2


DEFAULT_CONFIGS = (
    TrainingConfig(
        name="small",
        learning_rate=3e-4,             # tweak?
        weight_decay=1e-4,
        batch_size=32,
        epochs=30,                      # tweak?
        embedding_dim=128,
        hidden_dim=512,
        dropout=0.1,
        num_heads=4,
        num_layers=2,
    ),
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
        genes = GenomicsParquetDataset.parse_variant_sequence(row["variant_sequence"]) # read the patient's variant_sequence 
        token_ids = self.vocabulary.encode(genes, self.max_genes) # encodes into fixed-length token_ids
        input_token_ids = token_ids.clone()
        attention_mask = (token_ids != self.vocabulary.pad_index).long() # attention mask is created so that all non-pad positions are 1, pad positions are 0

        non_pad_positions = torch.nonzero(token_ids != self.vocabulary.pad_index, as_tuple=False).flatten() 
        if non_pad_positions.numel() == 0:
            raise ValueError("Encountered an empty mutation set after filtering.")

        known_gene_positions = torch.nonzero(
            (token_ids != self.vocabulary.pad_index)
            & (token_ids != self.vocabulary.unk_index)
            & (token_ids != self.vocabulary.mask_index),
            as_tuple=False,
        ).flatten()
        mask_source_positions = known_gene_positions
        if mask_source_positions.numel() == 0:
            mask_source_positions = non_pad_positions

        num_masked = min( 
            max(1, math.ceil(mask_source_positions.numel() * self.mask_probability)), # never mask padding, but mask at least 1 gene when possible
            int(mask_source_positions.numel()),
        )
        permutation = torch.randperm(mask_source_positions.numel())
        masked_positions = mask_source_positions[permutation[:num_masked]] 
        masked_token_ids = token_ids[masked_positions]
        input_token_ids[masked_positions] = self.vocabulary.mask_index # replace select positions in input_token_ids with [MASK] token

        target_multihot = torch.zeros(len(self.target_gene_ids), dtype=torch.float32) # mark only the masked genes as positive
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
    """Predicts masked genes from the encoder embedding."""

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


def split_dataframe(
    dataframe: pd.DataFrame,
    *,
    seed: int,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    patient_ids = list(pd.unique(dataframe["patient_id"]))
    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    n_total = len(patient_ids)
    n_train = int(n_total * train_fraction)
    n_val = int(n_total * val_fraction)
    train_ids = set(patient_ids[:n_train])
    val_ids = set(patient_ids[n_train:n_train + n_val])
    test_ids = set(patient_ids[n_train + n_val:])

    return (
        dataframe[dataframe["patient_id"].isin(train_ids)].reset_index(drop=True),
        dataframe[dataframe["patient_id"].isin(val_ids)].reset_index(drop=True),
        dataframe[dataframe["patient_id"].isin(test_ids)].reset_index(drop=True),
    )


def compute_train_stats(
    dataframe: pd.DataFrame,
    vocabulary: GeneVocabulary,
) -> dict[str, object]:
    
    gene_lists = [GenomicsParquetDataset.parse_variant_sequence(value) for value in dataframe["variant_sequence"]]
    gene_counts = [len(genes) for genes in gene_lists]
    flattened = [gene.lower() for genes in gene_lists for gene in genes]
    top_genes = pd.Series(flattened).value_counts().head(10).to_dict() if flattened else {}
    return {
        "num_train_rows": len(dataframe),
        "num_unique_train_patients": int(dataframe["patient_id"].nunique()) if "patient_id" in dataframe.columns else None,
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
    if model_name == "mean_pool":
        return DeepSetsEncoder(
            vocab_size=len(vocabulary),
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            output_dim=OUTPUT_DIM,
            pad_index=vocabulary.pad_index,
            pooling="mean",
            dropout=config.dropout,
        )
    return SetTransformerEncoder(
        vocab_size=len(vocabulary),
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        hidden_dim=config.hidden_dim,
        output_dim=OUTPUT_DIM,
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


def compute_pos_weight( # pos_weight changes loss so that positive labels are weighted far more
    dataframe: pd.DataFrame,
    vocabulary: GeneVocabulary,
) -> torch.Tensor:
    target_gene_ids = vocabulary.gene_token_ids()
    target_index_by_token_id = {
        token_id: idx for idx, token_id in enumerate(target_gene_ids)
    }
    positive_counts = torch.zeros(len(target_gene_ids), dtype=torch.float32)

    for value in dataframe["variant_sequence"]:
        genes = GenomicsParquetDataset.parse_variant_sequence(value)
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
    output_dir: Path,
) -> dict[str, object]:
    vocabulary = GeneVocabulary.build(
        (GenomicsParquetDataset.parse_variant_sequence(value) for value in train_df["variant_sequence"]),
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
    model = MaskedGenePredictor(encoder, output_dim=OUTPUT_DIM, vocabulary=vocabulary).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_val_recall_at_k = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None

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
            best_val_loss = val_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
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

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{model_name}_{config.name}_encoder.pt"
    torch.save(
        {
            "model_name": model_name,
            "config": asdict(config),
            "encoder_state_dict": model.encoder.state_dict(),
            "vocabulary_token_to_id": vocabulary.token_to_id,
            "pad_index": vocabulary.pad_index,
            "unk_index": vocabulary.unk_index,
            "mask_index": vocabulary.mask_index,
            "output_dim": OUTPUT_DIM,
        },
        checkpoint_path,
    )

    return {
        "config": asdict(config),
        "train_stats": train_stats,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_recall_at_k": best_val_recall_at_k,
        "test_loss": test_metrics["loss"],
        "test_recall_at_k": test_metrics["recall_at_k"],
        "vocabulary_size": len(vocabulary),
        "num_targets": len(target_gene_ids),
        "encoder_checkpoint": str(checkpoint_path),
    }


def save_json(data: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a genomics text encoder from metadata.parquet with 70/15/15 splits."
    )
    parser.add_argument("metadata_parquet", type=Path)
    parser.add_argument(
        "--model",
        choices=("mean_pool", "set_transformer"),
        default="set_transformer",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/genomics_text_encoder_runs"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    dataframe = GenomicsParquetDataset.load_parquet(args.metadata_parquet)
    filtered_df = GenomicsParquetDataset.load_filtered_dataframe(args.metadata_parquet)
    train_df, val_df, test_df = split_dataframe(filtered_df, seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs: list[dict[str, object]] = []
    results: dict[str, object] = {
        "model": args.model,
        "metadata_parquet": str(args.metadata_parquet),
        "seed": args.seed,
        "device": str(device),
        "num_rows_before_filtering": len(dataframe),
        "num_rows_after_filtering": len(filtered_df),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "runs": runs,
    }

    for config in DEFAULT_CONFIGS:
        runs.append(
            train_one_configuration(
                args.model,
                train_df,
                val_df,
                test_df,
                config=config,
                device=device,
                output_dir=args.output_dir,
            )
        )

    output_path = args.output_dir / f"{args.model}_results.json"
    save_json(results, output_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
