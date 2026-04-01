# Used Car Price Intelligence Platform

End-to-end machine learning project for used car price prediction, structured as a professional 3-stage workflow from analysis to refined modeling.

## Project Overview

This repository contains one connected pipeline across three stages:

- `01-eda`: exploratory data analysis, quality checks, and feature-target insights
- `02-baseline-modeling`: multi-model benchmark and best-model selection
- `03-model-refinement`: regularized refinement, tuning, and diagnostic evaluation

Dataset summary:

- Records: 500 
- Features: 29
- Target: `price`
- Shared dataset: `data/usedcars.csv`

## Business Problem

Used car valuation is challenging because market price is influenced by many non-linear factors (vehicle specs, condition proxies, and category-level effects). Incorrect pricing creates two major risks:

- Underpricing: loss of revenue/margin
- Overpricing: slower inventory movement

This project builds a data-driven pricing intelligence workflow to improve consistency, transparency, and predictive accuracy.

## Stage Summary

### Stage 01 - EDA

- Data quality checks: missing values, duplicates, dtypes
- Univariate and bivariate visual exploration
- Numerical and categorical behavior vs `price`
- Correlation and outlier-focused inspection
- Exported visuals in a structured output layout

### Stage 02 - Baseline Modeling

- Benchmarked model families:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - ElasticNet Regression
  - Polynomial Regression (degree 2 and 3)
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Evaluation with CV and test-set metrics (R2, RMSE, MAE)
- Best model persisted for reuse

### Stage 03 - Model Refinement

- Polynomial + Ridge pipeline with scaling
- Hyperparameter tuning using Grid Search
- 5-fold validation for stability checks
- Diagnostic plots:
  - Predicted vs Actual
  - Residual diagnostics (scatter + distribution)

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
|   |-- Used Cars Price Prediction.ipynb
|   `-- outputs/
|       |-- figures/
|       |-- metrics/
|       `-- models/
`-- 03-model-refinement/
    |-- README.md
    |-- OUTPUT_PREVIEWS.md
    |-- Used Car Price Prediction.ipynb
    `-- outputs/
        `-- figures/
```

## Methods and Evaluation

- Preprocessing: numeric conversion, one-hot encoding, scaling, missing-value handling
- Validation: train/test split + cross-validation
- Tuning: Grid Search for regularization control
- Metrics: R2, RMSE, MAE, residual diagnostics

## How to Run

1. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

2. Start Jupyter:

```bash
jupyter notebook
```

3. Run notebooks in order:

- `01-eda/EDA - UsedCars.ipynb`
- `02-baseline-modeling/Used Cars Price Prediction.ipynb`
- `03-model-refinement/Used Car Price Prediction.ipynb`

## Near-Term Next Steps

- Build ensemble strategies (Gradient Boosting + Random Forest blending/stacking)
- Add feature importance and explainability (permutation importance, SHAP)
- Package inference as a lightweight API
- Add monitoring and retraining workflow for production readiness

## Author

Visura Rodrigo
