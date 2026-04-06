from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "01-eda" / "outputs" / "processed" / "usedcars_stage1.csv"
STAGE4_METRICS_PATH = PROJECT_ROOT / "04-ensemble-modeling" / "outputs" / "metrics" / "best_ensemble_metrics.json"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"

MODEL_PATH = MODELS_DIR / "stage5_explainable_model.joblib"
METRICS_PATH = METRICS_DIR / "stage5_model_metrics.json"
IMPORTANCE_PATH = METRICS_DIR / "feature_importance.csv"
IMPORTANCE_PLOT_PATH = FIGURES_DIR / "feature_importance.png"
PREDICTION_PLOT_PATH = FIGURES_DIR / "prediction_scatter.png"

RANDOM_STATE = 42


def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_cleaned_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if "price" not in df.columns:
        raise ValueError("Expected target column 'price' not found in cleaned dataset.")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    features = df.drop(columns=["price"])
    target = df["price"].to_numpy()
    return features, target


def build_model() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    )


def train_stage5_model() -> tuple[GradientBoostingRegressor, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    df = load_cleaned_data()
    features, target = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    model = build_model()
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def compute_permutation_importance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error",
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return importance_df


def save_importance_plot(importance_df: pd.DataFrame, output_path: Path, top_n: int = 15) -> None:
    top_features = importance_df.head(top_n).sort_values("importance_mean", ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["importance_mean"], xerr=top_features["importance_std"])
    plt.title("Stage 05: Permutation Importance")
    plt.xlabel("Increase in RMSE when feature is shuffled")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_prediction_plot(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors="none")
    bounds = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(bounds, bounds, "r--", linewidth=2)
    plt.title("Stage 05: Predicted vs Actual")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
