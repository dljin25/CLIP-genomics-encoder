from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.tokenize_gene_dataset import (
    TokenizedGeneSetDataset,
    CANCER_TYPE,
    DATASET,
    PATIENT_ID,
    VARIANT_SEQUENCE,
)
from src.project_config import load_project_config, resolve_project_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_icgc_cancer_type(icgc_tsv_path: str | Path) -> dict[str, str]:
    ttype_frame = pd.read_csv(
        Path(icgc_tsv_path),
        sep="\t",
        compression="infer",
        usecols=["sample_id", "ttype"],
        dtype={"sample_id": "string", "ttype": "string"},
    )
    ttype_frame = ttype_frame.dropna(subset=["sample_id", "ttype"]).copy()
    ttype_frame["sample_id"] = ttype_frame["sample_id"].astype("string").str.strip()
    ttype_frame["ttype"] = ttype_frame["ttype"].astype("string").str.strip()
    ttype_frame = ttype_frame[
        (ttype_frame["sample_id"] != "") & (ttype_frame["ttype"] != "")
    ]
    unique_ttypes = ttype_frame.drop_duplicates(subset=["sample_id"])
    return dict(zip(unique_ttypes["sample_id"], unique_ttypes["ttype"]))


def build_gdc_parquet_dataframe(
    gdc_parquet_path: str | Path,
    *,
    gdc_cancer_type: str,
) -> pd.DataFrame:
    dataframe = pd.read_parquet(Path(gdc_parquet_path)).copy()
    processed = dataframe.loc[:, [PATIENT_ID, VARIANT_SEQUENCE]].copy()
    keep_mask = ~processed[VARIANT_SEQUENCE].apply(TokenizedGeneSetDataset.is_missing_variant_sequence)
    processed = processed.loc[keep_mask].copy()
    processed[PATIENT_ID] = processed[PATIENT_ID].astype("string").str.strip()
    processed = processed[processed[PATIENT_ID].notna() & (processed[PATIENT_ID] != "")]
    processed[DATASET] = "gdc"
    processed[CANCER_TYPE] = gdc_cancer_type
    return processed.reset_index(drop=True)


def build_icgc_parquet_dataframe(
    icgc_parquet_path: str | Path,
    icgc_tsv_path: str | Path,
    *,
    unknown_cancer_type: str,
) -> pd.DataFrame:
    dataframe = pd.read_parquet(Path(icgc_parquet_path)).copy()
    processed = dataframe.loc[:, [PATIENT_ID, VARIANT_SEQUENCE]].copy()
    keep_mask = ~processed[VARIANT_SEQUENCE].apply(TokenizedGeneSetDataset.is_missing_variant_sequence)
    processed = processed.loc[keep_mask].copy()
    processed[PATIENT_ID] = processed[PATIENT_ID].astype("string").str.strip()
    processed = processed[processed[PATIENT_ID].notna() & (processed[PATIENT_ID] != "")]
    processed[DATASET] = "icgc"
    processed[CANCER_TYPE] = processed[PATIENT_ID].map(load_icgc_cancer_type(icgc_tsv_path))
    processed[CANCER_TYPE] = processed[CANCER_TYPE].fillna(unknown_cancer_type).astype("string").str.strip()
    processed.loc[processed[CANCER_TYPE] == "", CANCER_TYPE] = unknown_cancer_type
    return processed.reset_index(drop=True)


def build_parquet(
    gdc_parquet_path: str | Path,
    icgc_parquet_path: str | Path,
    icgc_tsv_path: str | Path,
    *,
    gdc_cancer_type: str,
    unknown_cancer_type: str,
) -> pd.DataFrame:
    df = pd.concat( # pool parquet files
        [
            build_gdc_parquet_dataframe(gdc_parquet_path, gdc_cancer_type=gdc_cancer_type),
            build_icgc_parquet_dataframe(
                icgc_parquet_path,
                icgc_tsv_path,
                unknown_cancer_type=unknown_cancer_type,
            ),
        ],
        ignore_index=True,
    )
    df[PATIENT_ID] = df[PATIENT_ID].astype("string").str.strip()
    df[DATASET] = df[DATASET].astype("string").str.strip()
    df[CANCER_TYPE] = df[CANCER_TYPE].astype("string").str.strip()
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one pooled training parquet from raw mutation data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Project config file. Defaults to config.yaml in the repo root.",
    )

    preliminary_args, _ = parser.parse_known_args()
    project_config = load_project_config(preliminary_args.config)

    data_config = project_config["data"]
    raw_config = data_config["raw"]
    processed_config = data_config["processed"]
    label_config = data_config["labels"]

    parser.set_defaults(
        gdc_parquet=resolve_project_path(raw_config["gdc_parquet"]),
        icgc_parquet=resolve_project_path(raw_config["icgc_parquet"]),
        icgc_tsv=resolve_project_path(raw_config["icgc_tsv"]),
        output_parquet=resolve_project_path(processed_config["pooled_parquet"]),
    )

    parser.add_argument("--gdc-parquet", type=Path)
    parser.add_argument("--icgc-parquet", type=Path)
    parser.add_argument("--icgc-tsv", type=Path)
    parser.add_argument("--output-parquet", type=Path)

    args = parser.parse_args()

    pooled = build_parquet(
        gdc_parquet_path=args.gdc_parquet,
        icgc_parquet_path=args.icgc_parquet,
        icgc_tsv_path=args.icgc_tsv,
        gdc_cancer_type=label_config["gdc_cancer_type"],
        unknown_cancer_type=label_config["unknown_cancer_type"],
    )

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pooled.to_parquet(args.output_parquet, index=False)

    print(f"Wrote {len(pooled)} rows to {args.output_parquet}")
    print("Rows by dataset:")
    print(pooled[DATASET].value_counts().sort_index().to_string())
    print("\nRows by cancer type:")
    print(pooled[CANCER_TYPE].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
