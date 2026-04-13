from __future__ import annotations

import torch
import torch.nn as nn


class DeepSetsEncoder(nn.Module):
    """Permutation-invariant Deep Sets encoder for mutation sets."""

    def __init__(
        self,
        vocab_size: int,
        *,
        embedding_dim: int = 256,
        hidden_dim: int = 1024,
        output_dim: int = 512,
        pad_index: int = 0,
        pooling: str = "sum",
        dropout: float = 0.1,
    ) -> None:
        """Initialize the embedding, phi network, pooling, and rho network."""

        super().__init__()

        if pooling not in {"sum", "mean"}:
            raise ValueError("pooling must be either 'sum' or 'mean'.")

        self.pooling = pooling
        self.output_dim = output_dim
        self.pad_index = pad_index

        self.gene_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index,
        )
        self.phi = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.projection_head = self.rho

        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize embeddings and linear layers."""

        nn.init.normal_(self.gene_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.gene_embedding.weight[self.pad_index].zero_()

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def pool(
        self,
        set_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate per-gene features with masked sum or mean pooling."""

        mask = attention_mask.unsqueeze(-1).to(dtype=set_features.dtype)
        summed = (set_features * mask).sum(dim=1)
        if self.pooling == "sum":
            return summed
        counts = mask.sum(dim=1).clamp_min(1.0)
        return summed / counts

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a batch of tokenized gene sets."""

        return self.encode(token_ids, attention_mask)

    def encode(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply phi, permutation-invariant pooling, and rho projection."""

        if token_ids.ndim != 2:
            raise ValueError(
                f"Expected token_ids with shape [batch_size, num_genes], got {tuple(token_ids.shape)}."
            )

        if attention_mask is None:
            attention_mask = (token_ids != self.pad_index).long()
        elif attention_mask.shape != token_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as token_ids. "
                f"Got {tuple(attention_mask.shape)} and {tuple(token_ids.shape)}."
            )

        gene_embeddings = self.gene_embedding(token_ids)
        element_features = self.phi(gene_embeddings)
        pooled = self.pool(element_features, attention_mask)
        projected = self.rho(pooled)
        return projected
