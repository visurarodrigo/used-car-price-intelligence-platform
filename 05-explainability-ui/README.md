# Stage 05: Explainability

This folder is Stage 5 of the Used Car Price Intelligence Platform project.

## Purpose

Explain the Stage 04 prediction model and generate explainability artifacts.

## What This Stage Covers

- Retrains the winning Stage 04 model in the current environment for compatibility
- Generates permutation importance to explain feature influence
- Saves feature importance tables and diagnostic plots

## Files

- `stage5_explainability.py`: trains the Stage 05 model and generates explainability artifacts
- `stage5_shared.py`: shared paths, model, and artifact helpers

## Output Structure

Generated files are saved to:

- `outputs/metrics/stage5_model_metrics.json`
- `outputs/metrics/feature_importance.csv`
- `outputs/figures/feature_importance.png`
- `outputs/figures/prediction_scatter.png`
- `outputs/models/stage5_explainable_model.joblib`

## How to Run

1. Install dependencies:
   `pip install pandas numpy matplotlib scikit-learn joblib`
2. Generate the explainability artifacts:
   `python 05-explainability-ui/stage5_explainability.py`

## Author

Visura Rodrigo
