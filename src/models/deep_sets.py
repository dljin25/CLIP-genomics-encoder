from __future__ import annotations

import torch
import torch.nn as nn


class DeepSetsEncoder(nn.Module):
    """
    Basic Deep Sets encoder for binary genomics inputs.

    Expected input:
    - Tensor of shape (B, N) where N can be fixed or padded variable length.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        pad_index: int,
        dropout: float,
    ) -> None:
        """Initialize the embedding, phi network, and rho network."""

        super().__init__()

        self.output_dim = output_dim
        self.pad_index = pad_index

        self.gene_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index,
        )
        self.phi = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.projection_head = self.rho

        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize embeddings and linear layers."""

        nn.init.normal_(self.gene_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.gene_embedding.weight[self.pad_index].zero_() # zero out padding tokens

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.constant_(module.bias, 0)

    def pool(
        self,
        set_features: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate per-gene features with masked mean pooling."""

        mask = (token_ids != self.pad_index).unsqueeze(-1).to(dtype=set_features.dtype) # [batch_size, num_genes]
        summed = (set_features * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1.0)
        return summed / counts

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of tokenized gene sets."""

        return self.encode(token_ids)

    def encode(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply phi, permutation-invariant pooling, and rho projection."""

        x = self.gene_embedding(token_ids)
        x = self.phi(x)
        x = self.pool(x, token_ids)
        x = self.rho(x)
        return x
