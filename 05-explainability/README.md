# Stage 05: Explainability

This folder is Stage 05 of the Used Car Price Intelligence Platform project.

## Purpose

Explain the Stage 04 prediction model and generate explainability artifacts.

## What This Stage Covers

- Retrains the winning Stage 04 model in the current environment for compatibility
- Generates permutation importance to explain feature influence
- Saves feature importance tables and diagnostic plots

## Experiment Tracking (MLflow)

As with the previous modeling stages, **MLflow** is used to maintain a transparent record of the explainability process:
- **Automatic Logging**: Uses `mlflow.sklearn.autolog()` to capture the parameters and metrics of the explainable model.
- **Performance Comparison**: Logged metrics allow for a direct comparison between the "explainable" model and the high-performance ensemble from Stage 04.
- **Traceability**: Ensures that the feature importance results are tied to a specific model version and dataset split.

**To visualize the explainability results:**
Run the following command from the project root:
`mlflow ui --backend-store-uri sqlite:///mlflow.db`
Then open `http://127.0.0.1:5000` in your browser.

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
   `python 05-explainability/stage5_explainability.py`

## Author

Visura Rodrigo
