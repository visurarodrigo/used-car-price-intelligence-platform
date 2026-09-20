# Stage 04: Ensemble Modeling

This folder is Stage 4 of the Used Car Price Intelligence Platform project.

## Purpose

Build and evaluate ensemble strategies on top of the strongest Stage 02 models to improve prediction robustness and potentially reduce test error.

## What This Stage Covers

- Loads pre-cleaned data from Stage 01 (`01-eda/outputs/processed/usedcars_stage1.csv`)
- Trains tree-based base learners:
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Evaluates two ensemble strategies:
  - Weighted blending of Random Forest + Gradient Boosting
  - Stacking Regressor with Ridge meta-learner
- Compares all candidate models on test metrics (R2, RMSE, MAE)
- Saves best model and metrics artifacts
- Compares Stage 04 best RMSE against Stage 02 best RMSE

## Experiment Tracking (MLflow)

Consistent with previous stages, **MLflow** is used to track the ensemble experiments:
- **Automatic Tracking**: Uses `mlflow.sklearn.autolog()` to log hyperparameters and metrics for each base learner and ensemble model.
- **Strategy Comparison**: Each ensemble candidate (Blending, Stacking, Single Models) is logged as a separate run to allow side-by-side performance analysis.
- **Reproducibility**: The exact configuration of the ensemble (e.g., base learner counts, meta-learner type) is stored in the MLflow tracking server.

**To visualize the ensemble results:**
Run the following command from the project root:
`mlflow ui --backend-store-uri sqlite:///mlflow.db`
Then open `http://127.0.0.1:5000` in your browser.

## Files

- `stage4_ensemble_modeling.py`: Main script for ensemble training and evaluation
- `OUTPUT_PREVIEWS.md`: Visual output gallery with quick interpretation notes

## Output Structure

Generated files are saved to:

- `outputs/metrics/ensemble_comparison.csv`
- `outputs/metrics/best_ensemble_metrics.json`
- `outputs/models/best_ensemble_model.joblib`
- `outputs/figures/ensemble_predicted_vs_actual.png`

## How to Run

1. Install dependencies:
   `pip install pandas numpy matplotlib scikit-learn joblib`
2. Run stage script:
   `python 04-ensemble-modeling/stage4_ensemble_modeling.py`

## Author

Visura Rodrigo
