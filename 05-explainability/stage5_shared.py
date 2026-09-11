from __future__ import annotations
"""
Shared Utility Module for Stage 05: Model Explainability.

This module provides the core logic for training a regression model and
calculating feature importance using the Permutation Importance method.
It centralizes paths and helper functions used by the main explainability entry point.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Define the project root to maintain relative pathing across different environments
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Stage 05 depends on the cleaned Stage 01 dataset and Stage 04 performance metrics for comparison.
DATA_PATH = PROJECT_ROOT / "01-eda" / "outputs" / "processed" / "usedcars_stage1.csv"
STAGE4_METRICS_PATH = PROJECT_ROOT / "04-ensemble-modeling" / "outputs" / "metrics" / "best_ensemble_metrics.json"

# Output directory structure for explainability artifacts
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"

# Specific file paths for saved artifacts
MODEL_PATH = MODELS_DIR / "stage5_explainable_model.joblib"
METRICS_PATH = METRICS_DIR / "stage5_model_metrics.json"
IMPORTANCE_PATH = METRICS_DIR / "feature_importance.csv"
IMPORTANCE_PLOT_PATH = FIGURES_DIR / "feature_importance.png"
PREDICTION_PLOT_PATH = FIGURES_DIR / "prediction_scatter.png"

# Constant for reproducibility across splits and model initialization
RANDOM_STATE = 42


def ensure_output_dirs() -> None:
    """Creates the necessary directory structure for outputs if they do not already exist."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_cleaned_data() -> pd.DataFrame:
    """
    Loads the processed dataset from Stage 01.

    Raises:
        FileNotFoundError: If the cleaned dataset is missing.
        ValueError: If the target column 'price' is not found in the data.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if "price" not in df.columns:
        raise ValueError("Expected target column 'price' not found in cleaned dataset.")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Separates the dataset into a feature matrix (X) and the target vector (y).

    Args:
        df: The input DataFrame containing both features and the target.

    Returns:
        A tuple containing the feature DataFrame and the target numpy array.
    """
    features = df.drop(columns=["price"])
    target = df["price"].to_numpy()
    return features, target


def build_model() -> GradientBoostingRegressor:
    """
    Configures a Gradient Boosting Regressor.

    GBR is chosen because it is a powerful tree-based ensemble that provides
    stable results for permutation importance analysis.
    """
    return GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    )


def train_stage5_model() -> tuple[GradientBoostingRegressor, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Executes the full pipeline: data loading, splitting, and model training.

    Returns:
        The fitted model, training features, test features, training targets, and test targets.
    """
    df = load_cleaned_data()
    features, target = split_features_target(df)

    # 80/20 split to ensure the model is evaluated on unseen data
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
    """
    Computes standard regression metrics to assess model performance.

    Args:
        y_true: The actual prices.
        y_pred: The prices predicted by the model.

    Returns:
        A dictionary containing R², RMSE, and MAE.
    """
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
    """
    Calculates Permutation Importance for each feature.

    This technique shuffles the values of a single feature and observes how
    much the model's performance drops. If shuffling a feature doesn't
    increase the error (Importance = 0), the model does not rely on that feature.

    Args:
        model: The fitted regressor.
        X_test: The test feature set.
        y_test: The actual target values.
        feature_names: List of column names.

    Returns:
        A DataFrame containing mean importance and standard deviation per feature.
    """
    # scoring="neg_root_mean_squared_error" is used so that higher values are better
    # (scikit-learn convention), but the result is presented as the increase in error.
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
    """
    Generates a horizontal bar chart of the top N most important features.

    Args:
        importance_df: DataFrame with 'feature' and 'importance_mean'.
        output_path: Path where the PNG will be saved.
        top_n: Number of top features to display.
    """
    top_features = importance_df.head(top_n).sort_values("importance_mean", ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["importance_mean"], xerr=top_features["importance_std"])
    plt.title("Stage 05: Permutation Importance")
    plt.xlabel("Increase in RMSE when feature is shuffled")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_prediction_plot(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """
    Creates a scatter plot of Actual vs. Predicted prices.

    A perfect model would have all points on the diagonal red line.
    """
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
    """
    Writes a dictionary to a JSON file with consistent formatting.
    """
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
