from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

def load_df(tsv_path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(
        Path(tsv_path),
        sep="\t",
        compression="infer",
        dtype={"sample_id": "string", "ttype": "string", "gene": "string"},
    )
    return dataframe


def clean(gene_series: pd.Series) -> list[str]:
    cleaned_genes = [
        str(gene).strip().upper()
        for gene in gene_series
        if pd.notna(gene) and str(gene).strip()
    ]
    return sorted(set(cleaned_genes))


def build_patient_gene_table(df: pd.DataFrame) -> pd.DataFrame:
    working = df.loc[:, ["sample_id", "gene"]].copy()
    working = working.rename(columns={"sample_id": "patient_id"})
    working["patient_id"] = working["patient_id"].astype("string").str.strip()
    working["gene"] = working["gene"].astype("string").str.strip().str.upper()

    working = working[
        working["patient_id"].notna()
        & (working["patient_id"] != "")
        & working["gene"].notna()
        & (working["gene"] != "")
    ].copy()

    grouped = (
        working.groupby("patient_id", sort=True)["gene"]
        .apply(clean)
        .reset_index(name="gene_list")
    )

    grouped["variant_sequence"] = grouped["gene_list"].apply(str)
    result = grouped.loc[:, ["patient_id", "variant_sequence"]].copy()
    return result


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    working = df.loc[:, ["sample_id", "ttype"]].copy()
    working["sample_id"] = working["sample_id"].astype("string").str.strip()
    working["ttype"] = working["ttype"].astype("string").str.strip()

    working = working[
        working["sample_id"].notna()
        & (working["sample_id"] != "")
        & working["ttype"].notna()
        & (working["ttype"] != "")
    ].copy()

    return (
        working.groupby("ttype", sort=True)["sample_id"]
        .nunique()
        .reset_index(name="num_patients")
        .sort_values(["num_patients", "ttype"], ascending=[False, True])
        .reset_index(drop=True)
    )


def convert_tsv_to_parquet(
    input_tsv: str | Path,
    output_parquet: str | Path,
) -> pd.DataFrame:
    df = load_df(input_tsv)
    patient_gene_table = build_patient_gene_table(df)

    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patient_gene_table.to_parquet(output_path, index=False)
    return patient_gene_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert .tsv.gz file into a parquet with patient_id and variant_sequence."
        )
    )
    parser.add_argument("input_tsv", type=Path)
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("icgc.parquet"),
    )
    args = parser.parse_args()

    df = load_df(args.input_tsv)
    patient_gene_table = build_patient_gene_table(df)
    summary = summarize(df)

    output_path = Path(args.output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patient_gene_table.to_parquet(output_path, index=False)

    print("\nPatients per cancer type:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
