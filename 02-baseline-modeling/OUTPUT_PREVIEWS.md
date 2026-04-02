# Stage 02 Output Previews and Quick Interpretation

This document summarizes generated output files from Stage 02 and gives short interpretation notes.

Data context:
- Stage 2 now consumes the cleaned Stage 1 dataset (`01-eda/outputs/processed/usedcars_stage1.csv`)
- Cleaning, encoding, and missing-value handling are centralized in Stage 1
- Stage 2 focuses on fair model benchmarking and artifact generation

## 1) Model Comparison Chart

![Model Comparison](outputs/figures/model_comparison_bars.png)

Interpretation:
- The chart compares test performance across all trained models.
- Higher R2 and lower RMSE indicate better predictive quality.
- Use this chart as a quick ranking view before reviewing exact metrics in CSV/JSON outputs.

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
- Includes train/CV/test metrics for every model in one place.
- Use this file as the source of truth for model ranking and reproducibility.
- Compare CV and test scores together to catch overfitting.

## 4) Best Model Metrics Summary

File: [outputs/metrics/best_model_metrics.json](outputs/metrics/best_model_metrics.json)

Interpretation:
- Selected best model: Gradient Boosting
- Test R², RMSE, and MAE summarize final holdout performance.
- This JSON is the compact summary typically used by downstream stages and reports.

## 5) Saved Model Artifact

File: [outputs/models/best_model.joblib](outputs/models/best_model.joblib)

Interpretation:
- Serialized model ready for reuse in later stages.
- Load this file for inference without retraining.

## Practical Reading Guide

- Use R2 to estimate explained variance quality.
- Use RMSE and MAE to understand average error magnitude (in the same units as `price`).
- Prefer models with consistently strong cross-validation and test scores.
- Investigate outliers before deciding final production candidates.
