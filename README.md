# Genomics Encoder for CLIP (Masked Gene Pretraining)

This repository implements a **genomics encoder** designed to be paired with an image encoder in a CLIP-style multimodal model. The encoder learns meaningful representations of mutation sets using **self-supervised masked gene prediction**.



## Overview

The goal of this project is to learn a function:

mutation set → embedding

where the embedding captures:
- gene co-occurrence patterns  
- pathway-level structure  

These embeddings can later be aligned with medical imaging (e.g., CT scans) using **contrastive learning (CLIP)**.



## Objectives

We train the encoder using a **masked gene prediction objective**, inspired by language models:

> Given a partially masked mutation set, predict which genes were hidden.

Example:

**Input:**
[VHL, MASK, BAP1, MASK]

**Target:**
[PBRM1, SETD2] 

This forces the model to learn:
- dependencies between genes  
- latent biological structure



## Dataset Format

Input data is a parquet file with:

- `patient_id`
- `variant_sequence` (list of mutated genes)

Each patient is treated as a **set of genes**.



## Architecture

Note: Since we're dealing with a dataframe of patient_ids each with a list of mutated genes, our encoder needs to be permutation-invariant (the order of genes does not matter). 
      The primary ways to do this are Set Transformers (https://arxiv.org/pdf/1810.00825) and DeepSets (https://arxiv.org/abs/1703.06114). 

### 1. Set Transformer Encoder (Primary)

The main model uses a **Set Transformer** to encode mutation sets:

- uses attention to model **gene–gene interactions**  
- produces a fixed-size 512d embedding  


### 2. DeepSets Encoder

We also include a simpler **DeepSets-style encoder**:

- embeds each gene independently  
- averages (mean-pools) embeddings

Note: DeepSets underperforms the Set Transformer, indicating that **gene–gene interactions are important**.



## Training Objective

We use **multi-label binary cross-entropy**:

- output: logits over all genes  
- target: multi-hot vector of masked genes  

Loss:
- `BCEWithLogitsLoss`
- includes `pos_weight` to handle class imbalance  

---

## Evaluation

### Recall@K

> Fraction of masked genes recovered in the top-K predictions

Example:
- masked genes: `{A, B, C}`  
- top-10 predictions: `{A, D, E, ...}`  
- recall@10 = 1/3  



## Training Pipeline

1. Build gene vocabulary (with frequency filtering)
2. Convert gene sets → token IDs
3. Randomly mask a subset of genes
4. Train encoder to predict masked genes
5. Track validation recall@K
6. Save best encoder checkpoint



## Saving the Encoder

After training, we save:

- encoder weights (`.pt`)
- vocabulary (`.json`)
- config  

These are reused for CLIP training.

