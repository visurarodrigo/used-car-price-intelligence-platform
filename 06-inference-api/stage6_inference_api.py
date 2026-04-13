from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import GradientBoostingRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "05-explainability" / "outputs" / "models" / "stage5_explainable_model.joblib"
SCHEMA_PATH = PROJECT_ROOT / "01-eda" / "outputs" / "processed" / "usedcars_stage1.csv"
TARGET_COLUMN = "price"

app = FastAPI(title="Used Car Price Inference API", version="1.0.0")

MODEL: Any | None = None
FEATURE_COLUMNS: list[str] = []
MODEL_SOURCE = ""


class PredictRequest(BaseModel):
    features: dict[str, Any] = Field(
        ..., description="Feature-value mapping using Stage 1 cleaned feature names"
    )


class BatchPredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(
        ..., description="List of feature-value mappings using Stage 1 cleaned feature names"
    )


def _to_float(value: Any, feature_name: str) -> float:
    """Coerce incoming feature values to float for model compatibility."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Feature '{feature_name}' must be numeric. Got: {value!r}") from exc


def _prepare_row(features: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    if not FEATURE_COLUMNS:
        raise RuntimeError("Feature schema is not loaded.")

    row = {col: 0.0 for col in FEATURE_COLUMNS}
    unknown_features: list[str] = []

    for key, value in features.items():
        if key in row:
            row[key] = _to_float(value, key)
        else:
            unknown_features.append(key)

    vector = np.array([row[col] for col in FEATURE_COLUMNS], dtype=float).reshape(1, -1)
    return vector, unknown_features


def _load_runtime_artifacts() -> None:
    global MODEL
    global FEATURE_COLUMNS
    global MODEL_SOURCE

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema source not found: {SCHEMA_PATH}")

    schema_df = pd.read_csv(SCHEMA_PATH)

    if TARGET_COLUMN not in schema_df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in schema source.")

    feature_columns = [col for col in schema_df.columns if col != TARGET_COLUMN]

    try:
        model = joblib.load(MODEL_PATH)
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

    # Keep feature order aligned with the model when available.
    model_feature_names = getattr(model, "feature_names_in_", None)
    if model_feature_names is not None:
        feature_columns = [str(col) for col in model_feature_names]

    MODEL = model
    FEATURE_COLUMNS = feature_columns


@app.on_event("startup")
def startup_event() -> None:
    _load_runtime_artifacts()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "message": "Used Car Price Inference API is running.",
        "docs": "/docs",
        "health": "/health",
        "features": "/features",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    if MODEL is None or not FEATURE_COLUMNS:
        raise HTTPException(status_code=500, detail="Model or schema not loaded")

    return {
        "status": "ok",
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "model_source": MODEL_SOURCE,
        "feature_count": len(FEATURE_COLUMNS),
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


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        vector, unknown_features = _prepare_row(request.features)
        prediction = float(MODEL.predict(vector)[0])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {
        "predicted_price": prediction,
        "unknown_features_ignored": unknown_features,
    }


@app.post("/predict-batch")
def predict_batch(request: BatchPredictRequest) -> dict[str, Any]:
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if not request.rows:
        raise HTTPException(status_code=400, detail="rows must contain at least one item")

    vectors: list[np.ndarray] = []
    unknown_per_row: list[list[str]] = []

    try:
        for row in request.rows:
            vector, unknown = _prepare_row(row)
            vectors.append(vector)
            unknown_per_row.append(unknown)

        matrix = np.vstack(vectors)
        predictions = [float(value) for value in MODEL.predict(matrix)]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}") from exc

    return {
        "count": len(predictions),
        "predictions": predictions,
        "unknown_features_ignored": unknown_per_row,
    }
