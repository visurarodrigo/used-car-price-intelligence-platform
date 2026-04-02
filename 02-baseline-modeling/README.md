# Stage 02: Professional Regression Benchmark

This folder is **Stage 2** of the **Used Car Price Intelligence Platform** project.

## Purpose

Build a regression benchmark pipeline that compares multiple model families and selects the best candidate for downstream stages.

## What This Stage Covers

- Robust data loading and numeric conversion from the Stage 01 snapshot when available, otherwise `../data/usedcars.csv`
- Automated train/test split with 5-fold cross-validation
- Model comparison :
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - ElasticNet Regression
  - Polynomial Regression (degree 2)
  - Polynomial Regression (degree 3)
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Standardized evaluation with R2, RMSE, and MAE
- Automatic best-model selection
- Artifact persistence for plots, metrics, and serialized model
- Stage 01 handoff-aware loading keeps this notebook connected to the EDA stage

## Output Structure

All generated outputs are saved automatically to:

- `outputs/figures/`
- `outputs/metrics/`
- `outputs/models/`

## Files

- `Used Cars Price Prediction.ipynb` - Main Stage 02 benchmark notebook

## How to Run

1. Install dependencies:
   `pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter`
2. Open and run all cells in:
   `Used Cars Price Prediction.ipynb`

## Author

Visura Rodrigo
