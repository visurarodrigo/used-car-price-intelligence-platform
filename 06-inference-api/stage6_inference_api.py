from __future__ import annotations
"""Serve real-time predictions with schema-aligned validation and model fallback loading."""

import importlib.util
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import GradientBoostingRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Prefer Stage 08 deployed model, then fall back to Stage 05 explainable artifact.
MODEL_CANDIDATE_PATHS = [
    PROJECT_ROOT / "08-productionization" / "outputs" / "models" / "deployed_model.joblib",
    PROJECT_ROOT / "05-explainability" / "outputs" / "models" / "stage5_explainable_model.joblib",
]
SCHEMA_PATH = PROJECT_ROOT / "01-eda" / "outputs" / "processed" / "usedcars_stage1.csv"
VALIDATION_DIR = PROJECT_ROOT / "07-data-validation"
TARGET_COLUMN = "price"

STAGE7_MODULE_PATH = VALIDATION_DIR / "stage7_data_validation.py"

# Load Stage 07 validation helpers dynamically to avoid import-path coupling.
_STAGE7_SPEC = importlib.util.spec_from_file_location("stage7_data_validation", STAGE7_MODULE_PATH)
if _STAGE7_SPEC is None or _STAGE7_SPEC.loader is None:
    raise FileNotFoundError(f"Validation module not found: {STAGE7_MODULE_PATH}")

_STAGE7_MODULE = importlib.util.module_from_spec(_STAGE7_SPEC)
_STAGE7_SPEC.loader.exec_module(_STAGE7_MODULE)

load_validation_profile = _STAGE7_MODULE.load_validation_profile
validate_features = _STAGE7_MODULE.validate_features

app = FastAPI(title="Used Car Price Inference API", version="1.0.0")

MODEL: Any | None = None
FEATURE_COLUMNS: list[str] = []
MODEL_SOURCE = ""
LOADED_MODEL_PATH: Path | None = None
VALIDATION_PROFILE: dict[str, Any] = {}


class PredictRequest(BaseModel):
    # A single prediction payload aligned to Stage 01 cleaned feature names.
    features: dict[str, Any] = Field(
        ..., description="Feature-value mapping using Stage 1 cleaned feature names"
    )


class BatchPredictRequest(BaseModel):
    # Multiple rows in one call for simple client-side throughput gains.
    rows: list[dict[str, Any]] = Field(
        ..., description="List of feature-value mappings using Stage 1 cleaned feature names"
    )


class ValidateRequest(BaseModel):
    # Validation-only request for checking payload quality before prediction.
    features: dict[str, Any] = Field(
        ..., description="Feature-value mapping using Stage 1 cleaned feature names"
    )
    strict: bool = Field(
        default=False,
        description="When true, missing and unknown fields are treated as errors.",
    )


def _to_float(value: Any, feature_name: str) -> float:
    """Coerce incoming feature values to float for model compatibility."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Feature '{feature_name}' must be numeric. Got: {value!r}") from exc


def _prepare_row(features: dict[str, Any], *, strict: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Validate and align a payload into the exact model feature order.
    if not FEATURE_COLUMNS:
        raise RuntimeError("Feature schema is not loaded.")

    validation = validate_features(features, strict=strict)
    if not validation["is_valid"]:
        raise ValueError("; ".join(validation["errors"]))

    frame = pd.DataFrame([[validation["aligned_features"][col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    return frame, validation


def _load_runtime_artifacts() -> None:
    global MODEL
    global FEATURE_COLUMNS
    global MODEL_SOURCE
    global LOADED_MODEL_PATH
    global VALIDATION_PROFILE

    # Pick the first available model artifact based on configured precedence.
    model_path = next((path for path in MODEL_CANDIDATE_PATHS if path.exists()), None)
    if model_path is None:
        checked_paths = [str(path.relative_to(PROJECT_ROOT)) for path in MODEL_CANDIDATE_PATHS]
        raise FileNotFoundError(f"Model not found in any candidate path: {checked_paths}")

    # Profile-derived feature schema keeps API and validation behavior synchronized.
    VALIDATION_PROFILE = load_validation_profile()
    feature_columns = list(VALIDATION_PROFILE["features"])

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema source not found: {SCHEMA_PATH}")

    schema_df = pd.read_csv(SCHEMA_PATH)

    try:
        model = joblib.load(model_path)
        MODEL_SOURCE = "saved-artifact"
    except ModuleNotFoundError:
        # If pickle compatibility breaks across sklearn versions, retrain a compatible model.
        X = schema_df.drop(columns=[TARGET_COLUMN])
        y = schema_df[TARGET_COLUMN].to_numpy()
        model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
        model.fit(X, y)
        MODEL_SOURCE = "retrained-fallback"
        model_path = None

    # Keep feature order aligned with the model when available.
    model_feature_names = getattr(model, "feature_names_in_", None)
    if model_feature_names is not None:
        feature_columns = [str(col) for col in model_feature_names]

    MODEL = model
    FEATURE_COLUMNS = feature_columns
    LOADED_MODEL_PATH = model_path


@app.on_event("startup")
def startup_event() -> None:
    # Warm-load model and schema once when the API process starts.
    _load_runtime_artifacts()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "message": "Used Car Price Inference API is running.",
        "docs": "/docs",
        "health": "/health",
        "features": "/features",
        "validate": "/validate",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    if MODEL is None or not FEATURE_COLUMNS:
        raise HTTPException(status_code=500, detail="Model or schema not loaded")

    return {
        "status": "ok",
        "model_path": (
            str(LOADED_MODEL_PATH.relative_to(PROJECT_ROOT)) if LOADED_MODEL_PATH else "in-memory-retrained"
        ),
        "model_source": MODEL_SOURCE,
        "feature_count": len(FEATURE_COLUMNS),
        "validation_profile_path": "07-data-validation/outputs/metrics/data_validation_profile.json",
    }


@app.get("/features")
def get_features() -> dict[str, Any]:
    if not FEATURE_COLUMNS:
        raise HTTPException(status_code=500, detail="Feature schema not loaded")

    return {
        "target_column": TARGET_COLUMN,
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
    }


@app.post("/validate")
def validate(request: ValidateRequest) -> dict[str, Any]:
    try:
        report = validate_features(request.features, strict=request.strict)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return report


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        vector, validation = _prepare_row(request.features)
        prediction = float(MODEL.predict(vector)[0])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {
        "predicted_price": prediction,
        "validation": validation,
    }


@app.post("/predict-batch")
def predict_batch(request: BatchPredictRequest) -> dict[str, Any]:
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if not request.rows:
        raise HTTPException(status_code=400, detail="rows must contain at least one item")

    frames: list[pd.DataFrame] = []
    validation_reports: list[dict[str, Any]] = []

    try:
        # Validate and align each row independently, then score in one matrix call.
        for row in request.rows:
            vector, validation = _prepare_row(row)
            frames.append(vector)
            validation_reports.append(validation)

        matrix = pd.concat(frames, ignore_index=True)
        predictions = [float(value) for value in MODEL.predict(matrix)]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}") from exc

    return {
        "count": len(predictions),
        "predictions": predictions,
        "validation": validation_reports,
    }
