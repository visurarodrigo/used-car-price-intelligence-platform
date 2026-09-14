from __future__ import annotations
"""
Monitor model quality, decide retraining, and emit lightweight deployment assets.

This module implements a production gate for the used car price model. It monitors
performance against a baseline (from Stage 05) and detects feature drift using
mean-shift analysis. Depending on the results, it either promotes the existing
model or retrains it on the full dataset before exporting a final deployment artifact.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# --- Project Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "01-eda" / "outputs" / "processed" / "usedcars_stage1.csv"
STAGE5_MODEL_PATH = PROJECT_ROOT / "05-explainability" / "outputs" / "models" / "stage5_explainable_model.joblib"
STAGE5_METRICS_PATH = PROJECT_ROOT / "05-explainability" / "outputs" / "metrics" / "stage5_model_metrics.json"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"

MONITORING_REPORT_PATH = METRICS_DIR / "monitoring_report.json"
DEPLOYMENT_INFO_PATH = METRICS_DIR / "deployment_manifest.json"
DEPLOYED_MODEL_PATH = MODELS_DIR / "deployed_model.joblib"

# --- Constants & Alert Thresholds ---
RANDOM_STATE = 42
# Trigger retraining if RMSE increases by more than 5% compared to baseline
RMSE_ALERT_RATIO = 0.05
# Z-score threshold for flagging a feature as "shifted"
MEAN_SHIFT_ALERT_Z = 1.0
# Trigger retraining if more than 30% of features show significant mean shift
MEAN_SHIFT_ALERT_RATIO = 0.30


def ensure_output_dirs() -> None:
    """Create necessary output folders for metrics and model artifacts."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """
    Save a dictionary to a JSON file with indentation.

    Args:
        path: Destination file path.
        payload: Dictionary to be saved.
    """
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load the cleaned dataset and split it into features (X) and target (y).

    Returns:
        A tuple containing the features DataFrame and the target NumPy array.

    Raises:
        FileNotFoundError: If the cleaned dataset is missing.
        ValueError: If the target column 'price' is not found.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "price" not in df.columns:
        raise ValueError("Expected target column 'price' not found in cleaned dataset.")

    X = df.drop(columns=["price"])
    y = df["price"].to_numpy()
    return X, y


def build_model() -> GradientBoostingRegressor:
    """
    Initialize the GradientBoostingRegressor with production-grade hyperparameters.

    Returns:
        A configured GradientBoostingRegressor instance.
    """
    return GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    )


def evaluate(model: Any, X_test: pd.DataFrame, y_test: np.ndarray) -> dict[str, float]:
    """
    Calculate standard regression metrics for the provided model.

    Args:
        model: The trained model to evaluate.
        X_test: Test feature set.
        y_test: True target values.

    Returns:
        A dictionary containing R2, RMSE, and MAE.
    """
    y_pred = model.predict(X_test)
    return {
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }


def compute_mean_shift_ratio(X_train: pd.DataFrame, X_test: pd.DataFrame) -> dict[str, Any]:
    """
    Detect feature drift by comparing the mean of training and test sets via Z-scores.

    Args:
        X_train: Training feature set (baseline).
        X_test: Evaluation feature set (current).

    Returns:
        A dictionary containing the count of shifted features, the shift ratio,
        and a list of the top 10 most shifted features.
    """
    train_mean = X_train.mean(numeric_only=True)
    test_mean = X_test.mean(numeric_only=True)
    # Use a small epsilon to avoid division by zero in std
    train_std = X_train.std(numeric_only=True, ddof=0).replace(0.0, 1e-9)

    # Calculate Z-score: abs(mean_diff) / std
    z_shift = ((test_mean - train_mean).abs() / train_std).fillna(0.0)
    flagged = z_shift[z_shift > MEAN_SHIFT_ALERT_Z].sort_values(ascending=False)

    ratio = float(len(flagged) / len(z_shift)) if len(z_shift) else 0.0
    return {
        "feature_count": int(len(z_shift)),
        "shifted_feature_count": int(len(flagged)),
        "shifted_feature_ratio": ratio,
        "z_threshold": MEAN_SHIFT_ALERT_Z,
        "top_shifted_features": [
            {"feature": str(name), "z_shift": float(value)}
            for name, value in flagged.head(10).items()
        ],
    }


def load_stage5_baseline_rmse() -> float | None:
    """
    Load the RMSE achieved during Stage 05 to serve as the production baseline.

    Returns:
        The baseline RMSE as a float, or None if the metrics file is missing.
    """
    if not STAGE5_METRICS_PATH.exists():
        return None

    with STAGE5_METRICS_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rmse = payload.get("test_rmse")
    if rmse is None:
        return None

    return float(rmse)


def load_or_rebuild_stage5_model(X_train: pd.DataFrame, y_train: np.ndarray) -> tuple[Any, str]:
    """
    Load the model from Stage 05 or rebuild it if the artifact is missing.

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        A tuple containing the model instance and a string indicating the source.
    """
    if not STAGE5_MODEL_PATH.exists():
        model = build_model()
        model.fit(X_train, y_train)
        return model, "retrained-missing-stage5-model"

    try:
        model = joblib.load(STAGE5_MODEL_PATH)
        return model, "loaded-stage5-artifact"
    except ModuleNotFoundError:
        # Fallback in case of environment mismatch/missing dependencies in the joblib file
        model = build_model()
        model.fit(X_train, y_train)
        return model, "retrained-stage5-compatibility-fallback"


def choose_deployed_model(
    current_model: Any,
    current_metrics: dict[str, float],
    baseline_rmse: float,
    mean_shift: dict[str, Any],
    X_full: pd.DataFrame,
    y_full: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    """
    Decide whether to promote the current model or retrain on the full dataset.

    Retraining is triggered if:
    1. RMSE has degraded beyond the RMSE_ALERT_RATIO relative to the baseline.
    2. The ratio of drifted features exceeds MEAN_SHIFT_ALERT_RATIO.

    Args:
        current_model: The existing model instance.
        current_metrics: Performance metrics of the current model.
        baseline_rmse: The gold-standard RMSE from Stage 05.
        mean_shift: Results from the drift analysis.
        X_full: Full dataset features.
        y_full: Full dataset targets.

    Returns:
        A tuple containing the model to be deployed and a decision log dictionary.
    """
    rmse_alert_level = baseline_rmse * (1.0 + RMSE_ALERT_RATIO)
    rmse_degraded = current_metrics["rmse"] > rmse_alert_level
    shift_alert = mean_shift["shifted_feature_ratio"] > MEAN_SHIFT_ALERT_RATIO

    should_retrain = rmse_degraded or shift_alert
    reasons = []

    if rmse_degraded:
        reasons.append(
            f"RMSE {current_metrics['rmse']:.4f} exceeded alert threshold {rmse_alert_level:.4f}."
        )
    if shift_alert:
        reasons.append(
            "Feature-mean shift ratio exceeded threshold "
            f"({mean_shift['shifted_feature_ratio']:.2%} > {MEAN_SHIFT_ALERT_RATIO:.2%})."
        )
    if not reasons:
        reasons.append("Metrics and drift checks are within alert thresholds.")

    if should_retrain:
        # Retrain on full data to maximize model exposure to all available samples
        deployed_model = build_model()
        deployed_model.fit(X_full, y_full)
        source = "stage8-retrained-full-data"
    else:
        # Promote Stage 05 model as is
        deployed_model = current_model
        source = "stage5-model-promoted"

    decision = {
        "should_retrain": should_retrain,
        "decision_reasons": reasons,
        "baseline_rmse": baseline_rmse,
        "rmse_alert_ratio": RMSE_ALERT_RATIO,
        "rmse_alert_level": rmse_alert_level,
        "mean_shift_alert_ratio": MEAN_SHIFT_ALERT_RATIO,
        "deployed_model_source": source,
    }
    return deployed_model, decision


def main() -> int:
    """
    Main execution flow for Stage 08: Monitor -> Decide -> Deploy.

    Returns:
        0 on successful execution.
    """
    ensure_output_dirs()

    # 1. Load data and prepare a test split for the current health check
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    # 2. Load existing model and evaluate it
    current_model, current_model_source = load_or_rebuild_stage5_model(X_train, y_train)
    current_metrics = evaluate(current_model, X_test, y_test)

    # 3. Establish baseline for comparison
    baseline_rmse = load_stage5_baseline_rmse()
    if baseline_rmse is None:
        baseline_rmse = current_metrics["rmse"]

    # 4. Detect feature drift
    mean_shift = compute_mean_shift_ratio(X_train, X_test)

    # 5. Run the decision logic to choose the final model
    deployed_model, decision = choose_deployed_model(
        current_model=current_model,
        current_metrics=current_metrics,
        baseline_rmse=baseline_rmse,
        mean_shift=mean_shift,
        X_full=X,
        y_full=y,
    )

    # 6. Export the final model artifact
    joblib.dump(deployed_model, DEPLOYED_MODEL_PATH)

    # 7. Generate the monitoring report for auditability
    monitoring_report = {
        "stage": "Stage 08 - Productionization",
        "current_model_source": current_model_source,
        "current_metrics": current_metrics,
        "baseline_rmse": baseline_rmse,
        "performance_delta_vs_baseline_rmse": current_metrics["rmse"] - baseline_rmse,
        "mean_shift_monitor": mean_shift,
        "retraining_decision": decision,
        "deployed_model_path": DEPLOYED_MODEL_PATH.relative_to(PROJECT_ROOT).as_posix(),
    }

    # 8. Generate the deployment manifest for the API
    deployment_manifest = {
        "stage": "Stage 08 - Productionization",
        "deployed_model_path": DEPLOYED_MODEL_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "api_entrypoint": "06-inference-api/stage6_inference_api.py",
        "default_port": 8000,
        "deployment_note": "Use the root-level Dockerfile and GitHub Actions workflow for containerized deployment.",
    }

    save_json(MONITORING_REPORT_PATH, monitoring_report)
    save_json(DEPLOYMENT_INFO_PATH, deployment_manifest)

    print("Stage 08 productionization complete.")
    print(f"Monitoring report: {MONITORING_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Deployment manifest: {DEPLOYMENT_INFO_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Deployed model: {DEPLOYED_MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Retraining triggered: {decision['should_retrain']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
