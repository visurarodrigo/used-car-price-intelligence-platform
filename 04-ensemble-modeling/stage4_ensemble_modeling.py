from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "01-eda" / "outputs" / "processed" / "usedcars_stage1.csv"
STAGE2_BEST_METRICS = PROJECT_ROOT / "02-baseline-modeling" / "outputs" / "metrics" / "best_model_metrics.json"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"

RANDOM_STATE = 42


def ensure_output_dirs() -> None:
    # Create output folders once so all artifacts can be saved safely.
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def find_best_blend_weight(
    y_true: np.ndarray,
    rf_preds: np.ndarray,
    gbr_preds: np.ndarray,
) -> tuple[float, float]:
    # Grid-search the blend weight on out-of-fold predictions to avoid leakage.
    best_weight = 0.5
    best_rmse = float("inf")

    for weight in np.linspace(0.0, 1.0, 21):
        preds = weight * gbr_preds + (1.0 - weight) * rf_preds
        score = rmse(y_true, preds)
        if score < best_rmse:
            best_rmse = score
            best_weight = float(weight)

    return best_weight, best_rmse


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors="none")
    bounds = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(bounds, bounds, "r--", linewidth=2)
    plt.title("Stage 04 Ensemble: Predicted vs Actual")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> int:
    ensure_output_dirs()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "price" not in df.columns:
        raise ValueError("Expected target column 'price' not found in cleaned dataset.")

    X = df.drop(columns=["price"])
    y = df["price"].to_numpy()

    # Keep split settings stable for reproducible stage-to-stage comparisons.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    # Train strong tree-based base models first.
    rf = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    gbr = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    )

    rf.fit(X_train, y_train)
    gbr.fit(X_train, y_train)

    rf_pred_test = rf.predict(X_test)
    gbr_pred_test = gbr.predict(X_test)

    rf_metrics = evaluate(y_test, rf_pred_test)
    gbr_metrics = evaluate(y_test, gbr_pred_test)

    # Build out-of-fold predictions for honest blend-weight selection.
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rf_oof = np.zeros_like(y_train, dtype=float)
    gbr_oof = np.zeros_like(y_train, dtype=float)

    for train_idx, valid_idx in kf.split(X_train):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_valid = X_train.iloc[valid_idx]
        y_fold_train = y_train[train_idx]

        rf_fold = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        gbr_fold = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        )

        rf_fold.fit(X_fold_train, y_fold_train)
        gbr_fold.fit(X_fold_train, y_fold_train)

        rf_oof[valid_idx] = rf_fold.predict(X_fold_valid)
        gbr_oof[valid_idx] = gbr_fold.predict(X_fold_valid)

    best_weight, _ = find_best_blend_weight(y_train, rf_oof, gbr_oof)
    blend_pred_test = best_weight * gbr_pred_test + (1.0 - best_weight) * rf_pred_test
    blend_metrics = evaluate(y_test, blend_pred_test)

    # Stacking learns a meta-model over base model outputs.
    stacking_model = StackingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
            (
                "gbr",
                GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=RANDOM_STATE,
                ),
            ),
        ],
        final_estimator=Ridge(alpha=1.0),
        passthrough=False,
        n_jobs=-1,
        cv=5,
    )
    stacking_model.fit(X_train, y_train)
    stacking_pred_test = stacking_model.predict(X_test)
    stacking_metrics = evaluate(y_test, stacking_pred_test)

    models = {
        "Random Forest": (rf_metrics, rf),
        "Gradient Boosting": (gbr_metrics, gbr),
        "Weighted Blend (GBR/RF)": (blend_metrics, {"weight_gbr": best_weight}),
        "Stacking Regressor": (stacking_metrics, stacking_model),
    }

    comparison_rows = []
    for model_name, (metrics, _) in models.items():
        comparison_rows.append(
            {
                "model": model_name,
                "test_r2": metrics["r2"],
                "test_rmse": metrics["rmse"],
                "test_mae": metrics["mae"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values("test_rmse", ascending=True)
    comparison_path = METRICS_DIR / "ensemble_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    # Pick the winner using RMSE (lower is better).
    best_row = comparison_df.iloc[0]
    best_model_name = str(best_row["model"])

    best_payload = {
        "best_model": best_model_name,
        "test_r2": float(best_row["test_r2"]),
        "test_rmse": float(best_row["test_rmse"]),
        "test_mae": float(best_row["test_mae"]),
        "blend_weight_gbr": float(best_weight),
    }

    if STAGE2_BEST_METRICS.exists():
        with STAGE2_BEST_METRICS.open("r", encoding="utf-8") as f:
            stage2 = json.load(f)
        best_payload["stage2_best_model"] = stage2.get("best_model")
        best_payload["stage2_test_rmse"] = stage2.get("test_rmse")
        best_payload["rmse_delta_vs_stage2"] = float(best_row["test_rmse"]) - float(stage2.get("test_rmse", 0.0))

    best_json_path = METRICS_DIR / "best_ensemble_metrics.json"
    with best_json_path.open("w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2)

    # Persist the winning model in a single, stable artifact path.
    if best_model_name == "Random Forest":
        y_best = rf_pred_test
        joblib.dump(rf, MODELS_DIR / "best_ensemble_model.joblib")
    elif best_model_name == "Gradient Boosting":
        y_best = gbr_pred_test
        joblib.dump(gbr, MODELS_DIR / "best_ensemble_model.joblib")
    elif best_model_name == "Weighted Blend (GBR/RF)":
        y_best = blend_pred_test
        blend_bundle = {
            "rf": rf,
            "gbr": gbr,
            "weight_gbr": best_weight,
        }
        joblib.dump(blend_bundle, MODELS_DIR / "best_ensemble_model.joblib")
    else:
        y_best = stacking_pred_test
        joblib.dump(stacking_model, MODELS_DIR / "best_ensemble_model.joblib")

    plot_predictions(y_test, y_best, FIGURES_DIR / "ensemble_predicted_vs_actual.png")

    print("Stage 04 complete.")
    print(f"Saved comparison table: {comparison_path}")
    print(f"Saved best metrics: {best_json_path}")
    print(f"Saved best model: {MODELS_DIR / 'best_ensemble_model.joblib'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
