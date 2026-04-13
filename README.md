# Genomics Encoder for CLIP

This repository trains a genomics encoder from patient-level mutation sets using a masked gene retrieval objective.

## Data

Training reads `data/raw/metadata_genomics.parquet` by default. The full file is about 600MB, so it is not stored in normal Git. Keep it in external storage such as Google Drive, Hugging Face Datasets, S3, Zenodo, or Git LFS, then place or symlink it at `data/raw/metadata_genomics.parquet` after cloning.

The file is already tokenized:

- `patient_id`
- `gene_symbol_list`
- `gene_symbol_index_list`
- `variant_id_list`
- `variant_id_index_list`
- `pathways_altered_list`
- `pathways_altered_index_list`

`data/raw/metadata_genomics_head10.parquet` is a 10-row preview with the same schema and is tracked in Git for smoke tests.

The `gene_symbol_index_list` column is used directly as model input. Raw gene ids span `0..33982`, so the trainer shifts gene ids by `+1` internally to reserve input id `0` for padding and uses input id `33984` for the mask token. The prediction head still predicts the original raw gene ids.

## Model

The model is `deep_sets`. Its encoder follows the Deep Sets form:

```text
rho(sum(phi(gene_embedding)))
```

The sum aggregation keeps the encoder permutation-invariant.

## Objective

For each patient set, training masks a fraction of genes and predicts the missing raw gene ids with binary cross entropy over the gene vocabulary. The main retrieval metric is `recall@10`.

## Splits

Rows are patient-level samples. The default split is:

- 70% train
- 15% validation
- 15% test

The split is controlled by `runtime.seed` in `config.yaml`.

## Train

Run DeepSets from the repository root:

```bash
python src/training/train.py
```

Outputs:

- metrics: `outputs/runs/deep_sets_results.json`
- encoder checkpoint: `outputs/checkpoints/deep_sets_default_encoder.pt`

To use the preview parquet for a smoke test:

```bash
python src/training/train.py data/raw/metadata_genomics_head10.parquet
```

For Colab, upload `metadata_genomics.parquet` to Google Drive, mount Drive, and pass that file explicitly:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
python src/training/train.py /content/drive/MyDrive/metadata_genomics.parquet
```
