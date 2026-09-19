from __future__ import annotations
"""
Stage 05: Model Explainability and Feature Importance.
...
"""

import json
import mlflow
import mlflow.sklearn

import joblib
...

from stage5_shared import (
    IMPORTANCE_PATH,
    IMPORTANCE_PLOT_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PREDICTION_PLOT_PATH,
    RANDOM_STATE,
    STAGE4_METRICS_PATH,
    compute_permutation_importance,
    ensure_output_dirs,
    evaluate_predictions,
    save_importance_plot,
    save_json,
    save_prediction_plot,
    train_stage5_model,
)


def main() -> int:
    # Initialize necessary output directories to avoid FileNotFoundError when saving
    ensure_output_dirs()

    # Setup MLflow
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('used-car-price-intelligence')
    mlflow.sklearn.autolog()

    # Train a fresh Stage 05-compatible model in the current environment.
    # We use a model that supports permutation importance for transparency.
    model, _, X_test, _, y_test = train_stage5_model()

    # Score the model on the held-out test split before generating explanations.
    # This ensures the explainability results are based on unseen data (generalization).
    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)

    # Permutation importance shows which features matter most to the fitted model.
    # It shuffles each feature individually and measures the drop in model performance.
    # A drop of 0 means the feature does not contribute to the model's predictive power.
    importance_df = compute_permutation_importance(model, X_test, y_test, list(X_test.columns))

    # Persist the trained model and all explainability artifacts for reuse.
    joblib.dump(model, MODEL_PATH)
    importance_df.to_csv(IMPORTANCE_PATH, index=False)
    save_importance_plot(importance_df, IMPORTANCE_PLOT_PATH)
    save_prediction_plot(y_test, y_pred, PREDICTION_PLOT_PATH)

    # Construct a summary payload containing model performance and top drivers.
    payload = {
        "model": "Gradient Boosting Regressor",
        "random_state": RANDOM_STATE,
        "test_r2": metrics["r2"],
        "test_rmse": metrics["rmse"],
        "test_mae": metrics["mae"],
        "top_features": importance_df.head(10)["feature"].tolist(),
    }

    # Compare these results with the best model from Stage 04.
    # This helps determine if the explainable model is significantly worse than
    # the ensemble model in terms of accuracy.
    if STAGE4_METRICS_PATH.exists():
        with STAGE4_METRICS_PATH.open("r", encoding="utf-8") as f:
            stage4 = json.load(f)
        payload["stage4_best_model"] = stage4.get("best_model")
        payload["stage4_test_rmse"] = stage4.get("test_rmse")
        payload["rmse_delta_vs_stage4"] = float(metrics["rmse"]) - float(stage4.get("test_rmse", 0.0))

    save_json(METRICS_PATH, payload)

    # Log progress and output paths to the console.
    print("Stage 05 explainability complete.")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved feature importance: {IMPORTANCE_PATH}")
    print(f"Saved importance plot: {IMPORTANCE_PLOT_PATH}")
    print(f"Saved prediction plot: {PREDICTION_PLOT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
