from __future__ import annotations

import ast
import math
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.project_config import load_project_config
from src.data.build_gene_vocabulary import GeneVocabulary

PROJECT = load_project_config()
SCHEMA = PROJECT["data"]["schema"]
PATIENT_ID = SCHEMA["patient_id_column"]
VARIANT_SEQUENCE = SCHEMA["variant_sequence_column"]
DATASET = SCHEMA["dataset_column"]
CANCER_TYPE = SCHEMA["cancer_type_column"]
DEFAULT_MIN_FREQUENCY = PROJECT["training"]["min_frequency"]


class TokenizedGeneSetDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Dataset for mutation-set inputs stored in a processed parquet file."""

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
        self.dataframe = self.load_training_dataframe(self.parquet_path)

    @staticmethod
    def load_parquet(parquet_path: str | Path) -> pd.DataFrame:
        return pd.read_parquet(Path(parquet_path))

    @classmethod
    def load_training_dataframe(cls, parquet_path: str | Path) -> pd.DataFrame:
        dataframe = cls.load_parquet(parquet_path)

        keep_mask = ~dataframe[VARIANT_SEQUENCE].apply(cls.is_missing_variant_sequence)
        filtered = dataframe.loc[keep_mask].copy()
        filtered[PATIENT_ID] = filtered[PATIENT_ID].astype("string").str.strip()
        filtered[DATASET] = filtered[DATASET].astype("string").str.strip()
        filtered[CANCER_TYPE] = filtered[CANCER_TYPE].astype("string").str.strip()

        return filtered.reset_index(drop=True)

    @classmethod
    def build_vocabulary(
        cls,
        parquet_path: str | Path,
        *,
        min_frequency: int = DEFAULT_MIN_FREQUENCY,
    ) -> GeneVocabulary:
        dataframe = cls.load_training_dataframe(parquet_path)
        return GeneVocabulary.build(
            gene_sequences=(cls.parse_variant_sequence(value) for value in dataframe[VARIANT_SEQUENCE]),
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

        if TokenizedGeneSetDataset.is_missing_variant_sequence(value):
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
        genes = self.parse_variant_sequence(row[VARIANT_SEQUENCE])
        token_ids = self.vocabulary.encode(genes, self.max_genes)
        attention_mask = (token_ids != self.vocabulary.pad_index).long()
        return {
            PATIENT_ID: row[PATIENT_ID],
            DATASET: row[DATASET],
            CANCER_TYPE: row[CANCER_TYPE],
            "token_ids": token_ids,
            "attention_mask": attention_mask,
        }
