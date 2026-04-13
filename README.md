# Used Car Price Intelligence Platform

End-to-end machine learning project for used car price prediction, structured as a professional 7-stage workflow from analysis to inference API and data validation.

## Project Overview

This repository contains one connected workflow across seven stages:

- `01-eda`: exploratory data analysis, quality checks, and feature-target insights
- `02-baseline-modeling`: multi-model benchmark and best-model selection
- `03-model-refinement`: regularized refinement, tuning, and diagnostic evaluation
- `04-ensemble-modeling`: blending and stacking strategies for robust prediction
- `05-explainability`: permutation importance and explainability artifacts
- `06-inference-api`: FastAPI-based inference service for live predictions
- `07-data-validation`: reusable validation checks for incoming prediction data

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

### Stage 05 - Explainability

- Retrains the winning Stage 04 model in the current environment for compatibility
- Generates permutation importance to explain which features drive predictions
- Saves feature importance tables and diagnostic plots

### Stage 06 - Inference API

- Loads the saved Stage 05 model artifact at startup
- Exposes endpoints for health check, schema inspection, single prediction, and batch prediction
- Aligns input payloads to the Stage 1 cleaned feature schema for consistent inference
- Includes a root landing route at `/` so the base URL returns a useful JSON response instead of 404

### Stage 07 - Data Validation

- Builds a reusable validation profile from the cleaned Stage 1 dataset
- Checks payloads for missing fields, unknown fields, numeric coercion, and out-of-range values
- Feeds validation results into the inference API before prediction

## Current Best Result (Stage 05 Explainability)

- Best Model: Gradient Boosting
- Test R²: 0.9676
- Test RMSE: 1667.09
- Test MAE: 923.55

Reference improvement vs Stage 02 best RMSE: -174.70

Source: `05-explainability/outputs/metrics/stage5_model_metrics.json`

## Project Structure

```text
used-car-price-intelligence-platform/
|-- README.md
|-- data/
|   `-- usedcars.csv
|-- 01-eda/
|   |-- README.md
|   |-- OUTPUT_PREVIEWS.md
|   |-- stage1_eda.ipynb
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
    |-- OUTPUT_PREVIEWS.md
    |-- stage4_ensemble_modeling.py
    `-- outputs/
        |-- figures/
        |-- metrics/
        `-- models/
`-- 05-explainability/
  |-- README.md
  |-- OUTPUT_PREVIEWS.md
  |-- stage5_explainability.py
  |-- stage5_shared.py
  `-- outputs/
    |-- figures/
    |-- metrics/
    `-- models/
`-- 06-inference-api/
    |-- README.md
    |-- sample_predict_payload.json
    `-- stage6_inference_api.py
`-- 07-data-validation/
    |-- README.md
    `-- stage7_data_validation.py
```

## Methods and Evaluation

- Preprocessing: numeric conversion, one-hot encoding, scaling, missing-value handling
- Validation: train/test split + cross-validation
- Tuning: Grid Search for regularization control
- Ensembling: weighted blending and stacking
- Explainability: permutation importance for feature influence
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
[Stage 05: Explainability]
    ↓
validation profile + data checks
    ↓
best_model.joblib + metrics
  ↓
[Stage 06: Inference API]
  ↓
live prediction endpoint
```

## How to Run

1. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

For Stage 06 API usage:

```bash
pip install fastapi uvicorn
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
   - Execute Stage 5 (explainability) → saves importance outputs and model artifact
  - Execute Stage 7 (validation) → saves validation profile and sample report
  - Validate Stage 6 API → confirms model and validation layer load correctly

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

  Then run Stage 5 explainability script:
  - `python 05-explainability/stage5_explainability.py`

  Then run Stage 6 API:
  - `python -m uvicorn stage6_inference_api:app --reload --app-dir 06-inference-api`

  Then run Stage 7 validation profile generation:
   - `python 07-data-validation/stage7_data_validation.py`

  Then open one of these in the browser:
  - `http://127.0.0.1:8000/`
  - `http://127.0.0.1:8000/docs`
  - `http://127.0.0.1:8000/health`
  - `http://127.0.0.1:8000/features`


## Author

Visura Rodrigo
