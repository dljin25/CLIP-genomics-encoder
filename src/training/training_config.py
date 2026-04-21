from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for one DeepSets training run."""

    name: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    recall_at_ks: tuple[int, ...]
    mask_probability: float
    max_genes: int
    embedding_dim: int
    hidden_dim: int
    dropout: float
    pos_weight_max: float


def build_training_config(
    project_config: dict[str, Any],
) -> TrainingConfig:
    """Build DeepSets training settings from the project config."""

    training_config = project_config["training"]

    return TrainingConfig(
        name                =training_config["model_name"],
        learning_rate       =training_config["learning_rate"],
        weight_decay        =training_config["weight_decay"],
        batch_size          =training_config["batch_size"],
        epochs              =training_config["epochs"],
        recall_at_ks        =training_config.get("recall_at_ks"),
        mask_probability    =training_config["mask_probability"],
        max_genes           =training_config["max_genes"],
        embedding_dim       =training_config["embedding_dim"],
        hidden_dim          =training_config["hidden_dim"],
        dropout             =training_config["dropout"],
        pos_weight_max      =training_config.get("pos_weight_max", 100.0),
    )
