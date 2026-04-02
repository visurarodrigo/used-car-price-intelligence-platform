# Processed Dataset: usedcars_stage1.csv

This folder contains the canonical cleaned dataset produced by Stage 1.

## File

- `usedcars_stage1.csv`

## What this dataset is

`usedcars_stage1.csv` is the Stage 1 handoff dataset used by downstream modeling notebooks.
It is generated from the raw source file at `data/usedcars.csv` after running the Stage 1 notebook.

## Why it exists

The project uses one centralized cleaned dataset so that Stage 2 and Stage 3 train on the same prepared input.
This avoids duplicated cleaning logic across stages and keeps results reproducible.

## Cleaning applied in Stage 1

1. Load raw data from `data/usedcars.csv`.
2. Drop rows where the target `price` is missing.
3. One-hot encode categorical columns (`drop_first=True`).
4. Convert all columns to numeric types.
5. Fill remaining numeric missing values with column median.

## Current snapshot stats

- Rows: 500
- Columns: 69
- Missing values: 0
- Target column: `price`

## Used by

- `02-baseline-modeling/stage2_baseline_modeling.ipynb`
- `03-model-refinement/stage3_model_refinement.ipynb`

Both stages prefer this file as the primary data source.

## How to regenerate

From project root:

```bash
python run_all_stages.py
```

Or run Stage 1 only in Jupyter:

- `01-eda/stage1_eda.ipynb`

Running Stage 1 updates this CSV in place.
