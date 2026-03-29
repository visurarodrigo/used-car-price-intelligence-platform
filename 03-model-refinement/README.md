# Stage 03: Model Refinement and Evaluation

This folder is **Stage 3** of the full **Used Car Price Intelligence Platform** project.

## Purpose

Improve baseline models into a more reliable predictive solution by focusing on:
- Better generalization
- Hyperparameter tuning
- Validation quality
- Error analysis

## What This Stage Covers

- Data preprocessing and train/test split
- Baseline linear model reference
- Cross-validation for stability checks
- Polynomial feature expansion + Ridge regularization
- Grid Search to tune Ridge `alpha`
- Final evaluation with R-squared, MSE, and residual diagnostics

## Files

- `Used Car Price Prediction.ipynb` - Main refinement and evaluation notebook
- `Predicted vs Actual Car Prices.png` - Prediction fit visualization
- `Residual Plot.png` - Error pattern analysis

## Dataset

- Shared dataset path: `../data/usedcars.csv`
- Continues from the same data and feature context as Stages 1 and 2

## How to Run

1. Install dependencies:
    `pip install pandas numpy matplotlib seaborn scikit-learn jupyter`
2. Open and run:
    `Used Car Price Prediction.ipynb`

## Author

Visura Rodrigo




