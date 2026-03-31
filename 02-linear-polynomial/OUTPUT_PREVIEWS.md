# Stage 02 Output Previews and Quick Interpretation

This document summarizes generated output files from Stage 02 and gives short interpretation notes.

## 1) Model Comparison Chart

![Model Comparison](outputs/figures/model_comparison_bars.png)

Interpretation:
- The chart compares test performance across all trained models.
- Higher R2 and lower RMSE indicate better predictive quality.
- One polynomial model behaves as a catastrophic outlier, which can compress axis readability for other bars.

## 2) Best Model Diagnostics

![Best Model Diagnostics](outputs/figures/best_model_diagnostics.png)

Interpretation:
- Left panel checks how close predictions are to the ideal diagonal line.
- Right panel shows residual spread; tighter concentration near zero indicates better fit stability.
- This is useful for quickly checking bias and error distribution shape.

## 3) Full Metrics Table

File: [outputs/metrics/model_comparison.csv](outputs/metrics/model_comparison.csv)

Interpretation:
- Contains cross-validation and test metrics for each model.
- Top performers by test RMSE are:
  - Gradient Boosting: test RMSE about 2630.31, test R2 about 0.9435
  - Random Forest: test RMSE about 2849.56, test R2 about 0.9336
- Polynomial Regression degree 2 has extreme metric values, indicating numerical instability in this setup.

## 4) Best Model Metrics Summary

File: [outputs/metrics/best_model_metrics.json](outputs/metrics/best_model_metrics.json)

Interpretation:
- Selected best model: Gradient Boosting
- Test R2: 0.9434514878432829
- Test RMSE: 2630.3094846427352
- Test MAE: 1634.6021185130946

## 5) Saved Model Artifact

File: [outputs/models/best_model.joblib](outputs/models/best_model.joblib)

Interpretation:
- Serialized model ready for reuse in later stages.
- Load this file for inference without retraining.

## Practical Reading Guide

- Use R2 to estimate explained variance quality.
- Use RMSE and MAE to understand average error magnitude.
- Prefer models with consistently strong cross-validation and test scores.
- Investigate outliers before deciding final production candidates.
