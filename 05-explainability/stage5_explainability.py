from __future__ import annotations
"""Stage 05 entry point: train explainable model and export explanation artifacts."""

import json
from pathlib import Path

import joblib

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
    load_cleaned_data,
    save_importance_plot,
    save_json,
    save_prediction_plot,
    split_features_target,
    train_stage5_model,
)


def main() -> int:
    ensure_output_dirs()

    # Train a fresh Stage 05-compatible model in the current environment.
    model, X_train, X_test, y_train, y_test = train_stage5_model()

    # Score the model on the held-out test split before generating explanations.
    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)

    # Permutation importance shows which features matter most to the fitted model.
    importance_df = compute_permutation_importance(model, X_test, y_test, list(X_test.columns))

    # Persist the trained model and all explainability artifacts for reuse.
    joblib.dump(model, MODEL_PATH)
    importance_df.to_csv(IMPORTANCE_PATH, index=False)
    save_importance_plot(importance_df, IMPORTANCE_PLOT_PATH)
    save_prediction_plot(y_test, y_pred, PREDICTION_PLOT_PATH)

    payload = {
        "model": "Gradient Boosting Regressor",
        "random_state": RANDOM_STATE,
        "test_r2": metrics["r2"],
        "test_rmse": metrics["rmse"],
        "test_mae": metrics["mae"],
        "top_features": importance_df.head(10)["feature"].tolist(),
    }

    # Reuse Stage 04 metrics when available so the RMSE comparison is explicit.
    if STAGE4_METRICS_PATH.exists():
        with STAGE4_METRICS_PATH.open("r", encoding="utf-8") as f:
            stage4 = json.load(f)
        payload["stage4_best_model"] = stage4.get("best_model")
        payload["stage4_test_rmse"] = stage4.get("test_rmse")
        payload["rmse_delta_vs_stage4"] = float(metrics["rmse"]) - float(stage4.get("test_rmse", 0.0))

    save_json(METRICS_PATH, payload)

    print("Stage 05 explainability complete.")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved feature importance: {IMPORTANCE_PATH}")
    print(f"Saved importance plot: {IMPORTANCE_PLOT_PATH}")
    print(f"Saved prediction plot: {PREDICTION_PLOT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
