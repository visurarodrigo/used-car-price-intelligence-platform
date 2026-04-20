from __future__ import annotations
"""Monitor model quality, decide retraining, and emit lightweight deployment assets."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "01-eda" / "outputs" / "processed" / "usedcars_stage1.csv"
STAGE5_MODEL_PATH = PROJECT_ROOT / "05-explainability" / "outputs" / "models" / "stage5_explainable_model.joblib"
STAGE5_METRICS_PATH = PROJECT_ROOT / "05-explainability" / "outputs" / "metrics" / "stage5_model_metrics.json"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"
DEPLOYMENT_DIR = OUTPUT_DIR / "deployment"

MONITORING_REPORT_PATH = METRICS_DIR / "monitoring_report.json"
DEPLOYMENT_INFO_PATH = METRICS_DIR / "deployment_manifest.json"
DEPLOYED_MODEL_PATH = MODELS_DIR / "deployed_model.joblib"

RANDOM_STATE = 42
RMSE_ALERT_RATIO = 0.05
MEAN_SHIFT_ALERT_Z = 1.0
MEAN_SHIFT_ALERT_RATIO = 0.30


def ensure_output_dirs() -> None:
    # Create output folders before writing metrics, models, and deployment files.
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    # Load Stage 01 cleaned data as the shared source of truth for this stage.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "price" not in df.columns:
        raise ValueError("Expected target column 'price' not found in cleaned dataset.")

    X = df.drop(columns=["price"])
    y = df["price"].to_numpy()
    return X, y


def build_model() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    )


def evaluate(model: Any, X_test: pd.DataFrame, y_test: np.ndarray) -> dict[str, float]:
    # Evaluate with the same metrics used in earlier modeling stages.
    y_pred = model.predict(X_test)
    return {
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }


def compute_mean_shift_ratio(X_train: pd.DataFrame, X_test: pd.DataFrame) -> dict[str, Any]:
    # Use simple z-score mean shifts as a lightweight drift proxy.
    train_mean = X_train.mean(numeric_only=True)
    test_mean = X_test.mean(numeric_only=True)
    train_std = X_train.std(numeric_only=True, ddof=0).replace(0.0, 1e-9)

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
    # Baseline RMSE is used as the production alert anchor.
    if not STAGE5_METRICS_PATH.exists():
        return None

    with STAGE5_METRICS_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rmse = payload.get("test_rmse")
    if rmse is None:
        return None

    return float(rmse)


def load_or_rebuild_stage5_model(X_train: pd.DataFrame, y_train: np.ndarray) -> tuple[Any, str]:
    # Prefer loading the saved Stage 05 model; rebuild if artifact is missing/incompatible.
    if not STAGE5_MODEL_PATH.exists():
        model = build_model()
        model.fit(X_train, y_train)
        return model, "retrained-missing-stage5-model"

    try:
        model = joblib.load(STAGE5_MODEL_PATH)
        return model, "loaded-stage5-artifact"
    except ModuleNotFoundError:
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
    # Trigger retraining only when performance degrades or drift crosses thresholds.
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
        deployed_model = build_model()
        deployed_model.fit(X_full, y_full)
        source = "stage8-retrained-full-data"
    else:
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


def write_deployment_files() -> dict[str, str]:
    # Generate simple container/run assets to make local deployment straightforward.
    dockerfile_path = DEPLOYMENT_DIR / "Dockerfile"
    compose_path = DEPLOYMENT_DIR / "docker-compose.yml"
    run_ps1_path = DEPLOYMENT_DIR / "start_api.ps1"
    run_sh_path = DEPLOYMENT_DIR / "start_api.sh"

    dockerfile_path.write_text(
        """FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir pandas numpy scikit-learn joblib fastapi uvicorn

EXPOSE 8000

CMD ["uvicorn", "stage6_inference_api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "06-inference-api"]
""",
        encoding="utf-8",
    )

    compose_path.write_text(
        """version: '3.9'\nservices:\n  used-car-api:\n    build:\n      context: ../..\n      dockerfile: 08-productionization/outputs/deployment/Dockerfile\n    image: used-car-price-intelligence:latest\n    ports:\n      - \"8000:8000\"\n    restart: unless-stopped\n""",
        encoding="utf-8",
    )

    run_ps1_path.write_text(
        """python -m uvicorn stage6_inference_api:app --host 0.0.0.0 --port 8000 --app-dir 06-inference-api\n""",
        encoding="utf-8",
    )

    run_sh_path.write_text(
        """#!/usr/bin/env bash\npython -m uvicorn stage6_inference_api:app --host 0.0.0.0 --port 8000 --app-dir 06-inference-api\n""",
        encoding="utf-8",
    )

    return {
        "dockerfile": dockerfile_path.relative_to(PROJECT_ROOT).as_posix(),
        "docker_compose": compose_path.relative_to(PROJECT_ROOT).as_posix(),
        "run_script_powershell": run_ps1_path.relative_to(PROJECT_ROOT).as_posix(),
        "run_script_bash": run_sh_path.relative_to(PROJECT_ROOT).as_posix(),
    }


def main() -> int:
    # End-to-end Stage 08 flow: monitor, decide, deploy, and publish manifests.
    ensure_output_dirs()

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    current_model, current_model_source = load_or_rebuild_stage5_model(X_train, y_train)
    current_metrics = evaluate(current_model, X_test, y_test)

    baseline_rmse = load_stage5_baseline_rmse()
    if baseline_rmse is None:
        baseline_rmse = current_metrics["rmse"]

    mean_shift = compute_mean_shift_ratio(X_train, X_test)

    deployed_model, decision = choose_deployed_model(
        current_model=current_model,
        current_metrics=current_metrics,
        baseline_rmse=baseline_rmse,
        mean_shift=mean_shift,
        X_full=X,
        y_full=y,
    )

    joblib.dump(deployed_model, DEPLOYED_MODEL_PATH)
    deployment_files = write_deployment_files()

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

    deployment_manifest = {
        "stage": "Stage 08 - Productionization",
        "deployed_model_path": DEPLOYED_MODEL_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "deployment_files": deployment_files,
        "api_entrypoint": "06-inference-api/stage6_inference_api.py",
        "default_port": 8000,
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
