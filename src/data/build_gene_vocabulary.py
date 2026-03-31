from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import torch

from src.project_config import load_project_config

PROJECT = load_project_config()
DEFAULT_MIN_FREQUENCY = PROJECT["training"]["min_frequency"]


class GeneVocabulary:
    """Vocabulary for gene symbols with PAD, UNK, and MASK support."""

    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    MASK_TOKEN = "[MASK]"

    def __init__(self, token_to_id: dict[str, int]) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {index: token for token, index in token_to_id.items()}
        self.pad_index = token_to_id[self.PAD_TOKEN]
        self.unk_index = token_to_id[self.UNK_TOKEN]
        self.mask_index = token_to_id[self.MASK_TOKEN]

    @classmethod
    def build(
        cls,
        gene_sequences: Iterable[Sequence[str]],
        min_frequency: int = DEFAULT_MIN_FREQUENCY,
    ) -> "GeneVocabulary":
        counter: Counter[str] = Counter()
        for genes in gene_sequences:
            counter.update(g.upper() for g in genes)

        token_to_id = {
            cls.PAD_TOKEN: 0,
            cls.UNK_TOKEN: 1,
            cls.MASK_TOKEN: 2,
        }
        for token, frequency in sorted(counter.items()):
            if frequency >= min_frequency:
                token_to_id[token] = len(token_to_id)
        return cls(token_to_id)

    def encode(self, genes: Sequence[str], max_genes: int) -> torch.Tensor:
        clipped = [gene.upper() for gene in genes[:max_genes]]
        token_ids = [self.token_to_id.get(gene, self.unk_index) for gene in clipped]
        token_ids.extend([self.pad_index] * (max_genes - len(token_ids)))
        return torch.tensor(token_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def gene_token_ids(self) -> list[int]:
        special_ids = {self.pad_index, self.unk_index, self.mask_index}
        return [
            token_id
            for token_id in self.token_to_id.values()
            if token_id not in special_ids
        ]
