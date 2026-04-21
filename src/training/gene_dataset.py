from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

@dataclass(frozen=True)
class GeneIndexing:
    """Maps raw gene ids into model input ids with reserved padding and mask ids."""

    raw_vocab_size: int
    pad_index: int = 0
    mask_index: int = 1

    @property
    def input_vocab_size(self) -> int:
        """Return the embedding vocabulary size including pad and mask ids."""

        return self.raw_vocab_size + 2

    def to_input_ids(self, raw_gene_ids: torch.Tensor) -> torch.Tensor:
        """Shift raw gene ids so zero and one can be reserved."""

        return raw_gene_ids + 2


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
    targets = torch.stack([item["target_multihot"] for item in batch])
    return {
        "input_token_ids": input_token_ids,
        "target_multihot": targets,
    }
