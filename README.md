# Genomics Encoder for CLIP

This repository trains a genomics encoder for mutation sets using a self-supervised masked gene prediction objective.

## Repository Layout

```text
CLIP-genomics-encoder/
├── data/
│   ├── raw/
│   │   ├── gdc.parquet
│   │   ├── icgc.parquet
│   │   └── TableS3_panorama_driver_mutations_ICGC_samples.public.tsv.gz
│   ├── processed/
│   │   └── all.parquet
├── outputs/
│   ├── runs/
│   ├── checkpoints/
├── src/
│   ├── data/
│   │   ├── build_gene_vocabulary.py
│   │   ├── tokenize_gene_dataset.py
│   │   ├── build_parquet.py
│   │   ├── convert_tsv_to_parquet.py
│   ├── models/
│   │   ├── deep_sets.py
│   │   ├── set_transformer.py
│   ├── training/
│   │   ├── train.py
├── requirements.txt
```

## Workflow

Project defaults now live in `config.yaml`. Both parquet-building and training read from that file by default, and command-line flags override specific values when needed.

### 1. Build The Pooled Parquet

This step:
- drops rows with missing `variant_sequence`
- assigns `cancer_type = Kidney-RCC` to every GDC row
- enriches ICGC rows with `cancer_type` from the bundled `TableS3...` TSV
- pools both datasets into one processed parquet

Run:

```bash
python src/data/build_parquet.py
```

Default output:

`data/processed/all.parquet`

The processed parquet contains:
- `patient_id`
- `variant_sequence`
- `dataset`
- `cancer_type`

### 2. Train the encoder

Training now reads only the processed parquet. It does not load raw GDC or ICGC files directly.

Run:

```bash
python src/training/train.py
```

Or point to a custom processed parquet:

```bash
python src/training/train.py data/processed/all.parquet
```

Training:
- builds the gene vocabulary from the train split
- stratifies train/val/test by `cancer_type`
- uses patient-level splits
- writes metrics to `outputs/runs/`
- writes checkpoints to `outputs/checkpoints/`

## Models

Two mutation-set encoders are available:
- `src/models/set_transformer.py`
- `src/models/deep_sets.py`

## Notes

- `src/training/train.py` is the main training entrypoint.
- `src/data/build_parquet.py` is the main parquet-building entrypoint.
- Legacy top-level modules in `src/` are kept as thin compatibility shims during the refactor.
