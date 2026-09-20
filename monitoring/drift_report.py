"""
Phase 4 - Data drift monitoring with Evidently AI
Run from the repo ROOT:  python monitoring/drift_report.py
"""
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------- config
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "01-eda" / "outputs" / "processed" / "usedcars_stage1.csv"
OUT_DIR = ROOT / "monitoring"
HTML_PATH = OUT_DIR / "drift_report.html"
SUMMARY_PATH = OUT_DIR / "drift_summary.json"

TARGET = "price"
SEED = 42
PREFERRED_DRIFT_COLS = ["engine-size", "curb-weight", "horsepower"]
ALPHA = 0.05  # KS-test significance level


# ---------------------------------------------------------------- data
def load_reference():
    df = pd.read_csv(DATA_PATH)
    # Same split as the Stage 3 notebook -> reference = what the model trained on
    train_df, _ = train_test_split(df, test_size=0.1, random_state=1)
    reference = train_df.drop(columns=[TARGET]).reset_index(drop=True)
    return train_df, reference


def pick_drift_columns(train_df, reference, k=3):
    chosen = [c for c in PREFERRED_DRIFT_COLS if c in reference.columns]
    if len(chosen) == k:
        return chosen
    # Fallback: continuous features most correlated with price
    corr = train_df.corr(numeric_only=True)[TARGET].drop(TARGET).abs()
    extra = [
        c for c in corr.sort_values(ascending=False).index
        if reference[c].nunique() > 10 and c not in chosen
    ]
    return chosen + extra[: k - len(chosen)]


def simulate_production(reference, drift_cols):
    """Bootstrap a 'production' batch, then inject 3 kinds of real-world shift."""
    rng = np.random.default_rng(SEED)
    prod = reference.sample(n=len(reference), replace=True, random_state=SEED).reset_index(drop=True)
    prod = prod.astype({c: float for c in drift_cols})

    a, b, c = drift_cols
    # 1) scale shift (e.g. newer, larger engines)
    prod[a] = prod[a] * rng.uniform(1.10, 1.30, len(prod))
    # 2) location shift (e.g. heavier cars): +0.75 std on average
    prod[b] = prod[b] + rng.normal(0.75 * reference[b].std(), 0.25 * reference[b].std(), len(prod))
    # 3) spread shift (e.g. wider power range)
    prod[c] = prod[c] * rng.uniform(0.90, 1.35, len(prod))
    return prod


# ---------------------------------------------------------------- KS summary (independent of Evidently version)
def ks_summary(reference, production, drift_cols):
    rows = []
    for col in reference.columns:
        if not pd.api.types.is_numeric_dtype(reference[col]) or reference[col].nunique() <= 10:
            continue  # skip binary / one-hot / low-cardinality columns
        stat, p = ks_2samp(reference[col], production[col])
        rows.append({
            "feature": col,
            "ks_stat": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "drifted": bool(p < ALPHA),
            "injected": col in drift_cols,
        })
    return pd.DataFrame(rows).sort_values("ks_stat", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- Evidently report (new API, legacy fallback)
def build_evidently_report(reference, production, html_path):
    try:
        from evidently import Report, Dataset, DataDefinition
        from evidently.presets import DataDriftPreset, DataSummaryPreset
        new_api = True
    except ImportError:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        new_api = False

    if new_api:
        ref_ds = Dataset.from_pandas(reference, data_definition=DataDefinition())
        cur_ds = Dataset.from_pandas(production, data_definition=DataDefinition())
        report = Report([DataDriftPreset(), DataSummaryPreset()])
        snapshot = report.run(cur_ds, ref_ds)
        snapshot.save_html(str(html_path))
    else:
        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=reference, current_data=production)
        report.save_html(str(html_path))
    return "new" if new_api else "legacy"


# ---------------------------------------------------------------- MLflow logging (separate experiment)
def log_to_mlflow(summary, table):
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment("used-car-price-monitoring")
        with mlflow.start_run(run_name="Drift_Check_Simulated"):
            mlflow.set_tag("stage", "monitoring")
            mlflow.log_params({
                "reference_rows": summary["reference_rows"],
                "production_rows": summary["production_rows"],
                "injected_features": ",".join(summary["injected_features"]),
                "ks_alpha": ALPHA,
            })
            mlflow.log_metrics({
                "drift_share": summary["drift_share"],
                "n_drifted_features": summary["n_drifted_features"],
                "n_features_tested": summary["n_features_tested"],
            })
            for _, r in table.iterrows():
                name = r["feature"].replace(" ", "_")
                mlflow.log_metric(f"ks_stat__{name}", r["ks_stat"])
            mlflow.log_artifact(str(HTML_PATH))
            mlflow.log_artifact(str(SUMMARY_PATH))
        print("Logged to MLflow experiment 'used-car-price-monitoring'")
    except Exception as e:  # never let logging break the report
        print(f"MLflow logging skipped: {e}")


# ---------------------------------------------------------------- main
def main():
    train_df, reference = load_reference()
    drift_cols = pick_drift_columns(train_df, reference)
    production = simulate_production(reference, drift_cols)
    print(f"Reference rows: {len(reference)} | Production rows: {len(production)}")
    print(f"Injected drift into: {drift_cols}")

    table = ks_summary(reference, production, drift_cols)
    n_drifted = int(table["drifted"].sum())
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "reference_rows": len(reference),
        "production_rows": len(production),
        "injected_features": drift_cols,
        "n_features_tested": len(table),
        "n_drifted_features": n_drifted,
        "drift_share": round(n_drifted / max(len(table), 1), 4),
        "features": table.to_dict(orient="records"),
    }

    api = build_evidently_report(reference, production, HTML_PATH)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nEvidently API used: {api}")
    print(f"Drift report saved to {HTML_PATH}")
    print(f"Drift summary saved to {SUMMARY_PATH}")
    print(f"\nKS drift test (continuous features, alpha={ALPHA}):")
    print(table.to_string(index=False))
    print(f"\n{n_drifted} of {len(table)} continuous features drifted")

    log_to_mlflow(summary, table)


if __name__ == "__main__":
    main()