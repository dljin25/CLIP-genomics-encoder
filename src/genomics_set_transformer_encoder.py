from __future__ import annotations

import torch
import torch.nn as nn

class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        residual = x
        normalized = self.norm1(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = residual + attended
        x = x + self.mlp(self.norm2(x))
        return x


class PoolingByMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_seeds: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.randn(1, num_seeds, embed_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        queries = self.seed_vectors.expand(batch_size, -1, -1)
        pooled, _ = self.attention(
            queries,
            self.norm(x),
            self.norm(x),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return pooled


class SetTransformerEncoder(nn.Module):
    """
    Set Transformer encoder for mutation sets.

    gene symbols
    -> embedding lookup
    -> stacked self-attention blocks
    -> pooling by multihead attention
    -> MLP projection
    -> 512d embedding
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        embedding_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        hidden_dim: int = 1024,
        output_dim: int = 512,
        pad_index: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pad_index = pad_index
        self.output_dim = output_dim

        self.gene_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index,
        )
        self.encoder_layers = nn.ModuleList(
            [
                MultiheadSelfAttentionBlock(
                    embed_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.pool = PoolingByMultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            num_seeds=1,
            dropout=dropout,
        )
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.gene_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.gene_embedding.weight[self.pad_index].zero_()

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode(token_ids, attention_mask)

    def encode(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

        key_padding_mask = attention_mask == 0
        x = self.gene_embedding(token_ids)

        for layer in self.encoder_layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        pooled = self.pool(x, key_padding_mask=key_padding_mask).squeeze(1)
        projected = self.projection_head(pooled)
        return projected
