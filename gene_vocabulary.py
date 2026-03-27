from __future__ import annotations

import ast
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset


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
        min_frequency: int = 3,
    ) -> "GeneVocabulary":
        counter: Counter[str] = Counter()
        for genes in gene_sequences:
            counter.update(g.upper() for g in genes)

        token_to_id = {
            cls.PAD_TOKEN: 0,
            cls.UNK_TOKEN: 1,
            cls.MASK_TOKEN: 2,
        }
        for token, frequency in sorted(counter.items()): # min_frequency = 3, so genes seen only once or twice in training become [UNK]
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


class GenomicsParquetDataset(Dataset[dict[str, torch.Tensor | str]]):
    """
    Dataset for mutation-set inputs stored in a metadata parquet file.

    Rows with null `variant_sequence` are dropped because they indicate missing mutation data.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        vocabulary: GeneVocabulary,
        *,
        max_genes: int = 256,
    ) -> None:
        self.parquet_path = Path(parquet_path)
        self.vocabulary = vocabulary
        self.max_genes = max_genes
        self.dataframe = self.load_filtered_dataframe(self.parquet_path)

    @staticmethod
    def load_parquet(parquet_path: str | Path) -> pd.DataFrame:
        return pd.read_parquet(Path(parquet_path))

    @classmethod
    def load_filtered_dataframe(cls, parquet_path: str | Path) -> pd.DataFrame:
        dataframe = cls.load_parquet(parquet_path)
        keep_mask = ~dataframe["variant_sequence"].apply(cls.is_missing_variant_sequence)
        return dataframe.loc[keep_mask].reset_index(drop=True)

    @classmethod
    def build_vocabulary(
        cls,
        parquet_path: str | Path,
        *,
        min_frequency: int = 3,
    ) -> GeneVocabulary:
        dataframe = cls.load_filtered_dataframe(parquet_path)
        return GeneVocabulary.build(
            gene_sequences=(cls.parse_variant_sequence(value) for value in dataframe["variant_sequence"]),
            min_frequency=min_frequency,
        )

    @staticmethod
    def is_missing_variant_sequence(value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, float):
            return math.isnan(value)
        if isinstance(value, str):
            return value.strip().lower() in {"", "null", "none", "nan"}
        return False

    @staticmethod
    def parse_variant_sequence(value: object) -> list[str]:
        if isinstance(value, (list, tuple)):
            return [str(item).strip().upper() for item in value if str(item).strip()]

        if GenomicsParquetDataset.is_missing_variant_sequence(value):
            return []

        if not isinstance(value, str):
            raise ValueError("Expected `variant_sequence` to be a stringified list, list, or null.")

        try:
            parsed = ast.literal_eval(value.strip())
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                "Expected `variant_sequence` to be a stringified Python list of gene symbols."
            ) from exc

        if not isinstance(parsed, (list, tuple)):
            raise ValueError("Expected parsed `variant_sequence` to be a list or tuple.")

        return [str(item).strip().upper() for item in parsed if str(item).strip()]

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.dataframe.iloc[index]
        genes = self.parse_variant_sequence(row["variant_sequence"])
        token_ids = self.vocabulary.encode(genes, self.max_genes)
        attention_mask = (token_ids != self.vocabulary.pad_index).long()
        return {
            "patient_id": row.get("patient_id", row.get("series_id", str(index))),
            "token_ids": token_ids,
            "attention_mask": attention_mask,
        }
