# Used Car Price Intelligence Platform

End-to-end machine learning project for used car price prediction, structured as a professional 3-stage workflow from analysis to refined modeling.

## Project Overview

This repository contains one connected pipeline across four stages:

- `01-eda`: exploratory data analysis, quality checks, and feature-target insights
- `02-baseline-modeling`: multi-model benchmark and best-model selection
- `03-model-refinement`: regularized refinement, tuning, and diagnostic evaluation
- `04-ensemble-modeling`: blending and stacking strategies for robust prediction

Dataset summary:

- Raw Records: 500 
- Raw Features: 29
- Target: `price`
- Raw dataset: `data/usedcars.csv`
- **Cleaned dataset** (after Stage 1): `01-eda/outputs/processed/usedcars_stage1.csv` (69 columns after one-hot encoding)

## Business Problem

Used car valuation is challenging because market price is influenced by many non-linear factors (vehicle specs, condition proxies, and category-level effects). Incorrect pricing creates two major risks:

- Underpricing: loss of revenue/margin
- Overpricing: slower inventory movement

This project builds a data-driven pricing intelligence workflow to improve consistency, transparency, and predictive accuracy.

## Stage Summary

### Stage 01 - EDA + Data Cleaning

- Data quality checks: missing values, duplicates, dtypes
- Univariate and bivariate visual exploration
- Numerical and categorical behavior vs `price`
- Correlation and outlier-focused inspection
- Exported visuals in a structured output layout
- **Full data cleaning pipeline**:
  - One-hot encodes categorical features
  - Removes rows with missing target
  - Converts all features to numeric
  - Imputes missing values with median
- Publishes fully cleaned, ready-to-model dataset at `01-eda/outputs/processed/usedcars_stage1.csv`

### Stage 02 - Baseline Modeling

- Loads pre-cleaned data from Stage 01 (all numeric, no missing values)
- Automated train/test split (80/20) with 5-fold cross-validation
- Benchmarked model families:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - ElasticNet Regression
  - Polynomial Regression (degree 2 and 3)
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Evaluation with CV and test-set metrics (R², RMSE, MAE)
- Best model persisted for reuse
- Preprocessing: scaling for linear models (no imputation—data already clean)
- Saves model, metrics, and comparison tables under `02-baseline-modeling/outputs/`

### Stage 03 - Model Refinement

- Loads pre-cleaned data from Stage 01 (same centralized dataset as Stage 02)
- Baseline Linear Regression for reference
- Polynomial feature expansion (degree 2) + Ridge regression
- Hyperparameter tuning using Grid Search on Ridge `alpha`
- 5-fold cross-validation for stability checks
- Preprocessing: scaling within cross-validation pipeline (prevents data leakage)
- Diagnostic plots:
  - Predicted vs Actual
  - Residual diagnostics (scatter + distribution)
- Loads Stage 02 baseline metrics for performance comparison

### Stage 04 - Ensemble Modeling

- Loads pre-cleaned data from Stage 01 (same centralized dataset as prior stages)
- Trains and evaluates strong tree-based candidates:
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Implements ensemble strategies:
  - Weighted blending (Gradient Boosting + Random Forest)
  - Stacking regressor with Ridge meta-learner
- Selects the best Stage 04 candidate by test RMSE
- Persists comparison metrics, best-model artifact, and prediction diagnostic figure
- Compares Stage 04 best RMSE against Stage 02 best benchmark

## Current Best Result (Stage 02 Benchmark)

- Best Model: Gradient Boosting
- Test R²: 0.9605 
- Test RMSE: 1841.79 
- Test MAE: 1082.22 

Source: `02-baseline-modeling/outputs/metrics/best_model_metrics.json`

## Project Structure

```text
used-car-price-intelligence-platform/
|-- README.md
|-- data/
|   `-- usedcars.csv
|-- 01-eda/
|   |-- README.md
|   |-- OUTPUT_PREVIEWS.md
|   |-- EDA - UsedCars.ipynb
|   `-- outputs/
|       |-- overview/
|       |-- numerical/
|       |-- categorical/
|       `-- bivariate/
|-- 02-baseline-modeling/
|   |-- README.md
|   |-- OUTPUT_PREVIEWS.md
|   |-- stage2_baseline_modeling.ipynb
|   `-- outputs/
|       |-- figures/
|       |-- metrics/
|       `-- models/
`-- 03-model-refinement/
    |-- README.md
    |-- OUTPUT_PREVIEWS.md
  |-- stage3_model_refinement.ipynb
    `-- outputs/
        `-- figures/
`-- 04-ensemble-modeling/
    |-- README.md
    |-- stage4_ensemble_modeling.py
    `-- outputs/
        |-- figures/
        |-- metrics/
        `-- models/
```

## Methods and Evaluation

- Preprocessing: numeric conversion, one-hot encoding, scaling, missing-value handling
- Validation: train/test split + cross-validation
- Tuning: Grid Search for regularization control
- Ensembling: weighted blending and stacking
- Metrics: R2, RMSE, MAE, residual diagnostics

## Data Flow

```
raw data (usedcars.csv)
    ↓
[Stage 01: EDA + Cleaning]
    ↓
cleaned data (usedcars_stage1.csv)
    ↓
[Stage 02: Baseline Modeling] ← [Stage 03: Model Refinement]
  ↓
[Stage 04: Ensemble Modeling]
    ↓
best_model.joblib + metrics
```

## How to Run

1. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

2. Run all stages with one command:

```bash
python run_all_stages.py
```

   This will:
   - Execute Stage 1 (EDA + cleaning) → outputs `usedcars_stage1.csv`
   - Execute Stage 2 (benchmarking) → uses cleaned data
   - Execute Stage 3 (refinement) → uses cleaned data
   - Execute Stage 4 (ensemble modeling) → saves ensemble metrics, model, and figure

3. Or run manually in Jupyter:

```bash
jupyter notebook
```

   Then execute notebooks in order:
   - `01-eda/stage1_eda.ipynb`
   - `02-baseline-modeling/stage2_baseline_modeling.ipynb`
   - `03-model-refinement/stage3_model_refinement.ipynb`

  Then run Stage 4 script:
  - `python 04-ensemble-modeling/stage4_ensemble_modeling.py`

## Near-Term Next Steps

- Expand and tune ensemble models to further improve prediction robustness
- Add model explainability to show which features drive predicted price
- Build a lightweight prediction UI for interactive price estimation

## Author

Visura Rodrigo
