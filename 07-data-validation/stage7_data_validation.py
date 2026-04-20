from __future__ import annotations
"""Build and apply reusable feature validation rules for inference payloads."""

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "01-eda" / "outputs" / "processed" / "usedcars_stage1.csv"
SAMPLE_PAYLOAD_PATH = PROJECT_ROOT / "06-inference-api" / "sample_predict_payload.json"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
PROFILE_PATH = METRICS_DIR / "data_validation_profile.json"
SAMPLE_REPORT_PATH = METRICS_DIR / "sample_validation_report.json"

TARGET_COLUMN = "price"
MISSING_FEATURE_RATIO_WARNING = 0.35


def ensure_output_dirs() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def load_cleaned_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' not found in cleaned dataset.")
    return df


def split_features(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column != TARGET_COLUMN]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Series, pd.Index)):
        return value.tolist()
    return value


def numeric_summary(series: pd.Series) -> dict[str, Any]:
    clean_series = pd.to_numeric(series, errors="coerce").dropna()
    if clean_series.empty:
        raise ValueError(f"Column '{series.name}' does not contain valid numeric values.")

    unique_values = sorted({float(value) for value in clean_series.unique()})
    return {
        "min": float(clean_series.min()),
        "max": float(clean_series.max()),
        "median": float(clean_series.median()),
        "mean": float(clean_series.mean()),
        "std": float(clean_series.std(ddof=0)) if len(clean_series) > 1 else 0.0,
        "unique_value_count": int(clean_series.nunique()),
        "is_binary": unique_values in ([0.0], [1.0], [0.0, 1.0]),
    }


@lru_cache(maxsize=1)
def load_validation_profile() -> dict[str, Any]:
    # Cache profile generation because the schema is static within one run.
    df = load_cleaned_data()
    feature_names = split_features(df)

    profile = {
        "schema_source": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "target_column": TARGET_COLUMN,
        "feature_count": len(feature_names),
        "features": feature_names,
        "feature_stats": {
            feature: numeric_summary(df[feature]) for feature in feature_names
        },
    }
    return profile


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _coerce_numeric(value: Any, feature_name: str) -> float:
    # Enforce numeric finite values so downstream model math is safe.
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Feature '{feature_name}' must be numeric. Got: {value!r}") from exc

    if not math.isfinite(numeric_value):
        raise ValueError(f"Feature '{feature_name}' must be a finite number. Got: {value!r}")

    return numeric_value


def validate_features(features: dict[str, Any], *, strict: bool = False) -> dict[str, Any]:
    # Validate, coerce, and align incoming features to the full training schema.
    profile = load_validation_profile()
    expected_features = profile["features"]
    feature_stats = profile["feature_stats"]

    aligned_features = {feature: 0.0 for feature in expected_features}
    unknown_features: list[str] = []
    missing_features: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    validated_features: dict[str, float] = {}

    for feature_name, value in features.items():
        # Unknown fields are warnings by default and errors in strict mode.
        if feature_name not in aligned_features:
            unknown_features.append(feature_name)
            message = f"Unknown feature '{feature_name}' ignored."
            if strict:
                errors.append(message)
            else:
                warnings.append(message)
            continue

        numeric_value = _coerce_numeric(value, feature_name)
        validated_features[feature_name] = numeric_value
        aligned_features[feature_name] = numeric_value

        stat = feature_stats[feature_name]
        # Flag values outside observed training ranges without hard-failing.
        if numeric_value < stat["min"] or numeric_value > stat["max"]:
            warnings.append(
                f"Feature '{feature_name}' is outside the observed training range "
                f"[{stat['min']}, {stat['max']}]."
            )

        if stat["is_binary"] and numeric_value not in (0.0, 1.0):
            warnings.append(f"Feature '{feature_name}' is binary in training data but received {numeric_value}.")

    for feature_name in expected_features:
        # Missing fields are defaulted to 0.0 to keep the API shape stable.
        if feature_name not in features:
            missing_features.append(feature_name)

    missing_ratio = len(missing_features) / len(expected_features) if expected_features else 0.0
    if missing_features:
        message = (
            f"{len(missing_features)} expected features were not supplied and were defaulted to 0.0."
        )
        if strict:
            errors.append(message)
        else:
            warnings.append(message)

    if missing_ratio >= MISSING_FEATURE_RATIO_WARNING:
        warnings.append(
            f"{len(missing_features)} of {len(expected_features)} features were missing "
            f"({missing_ratio:.0%}); the payload is sparse relative to the training schema."
        )

    result = {
        "is_valid": len(errors) == 0,
        "strict": strict,
        "provided_feature_count": len(features),
        "validated_feature_count": len(validated_features),
        "expected_feature_count": len(expected_features),
        "missing_feature_count": len(missing_features),
        "unknown_feature_count": len(unknown_features),
        "missing_features": missing_features,
        "unknown_features": unknown_features,
        "errors": errors,
        "warnings": warnings,
        "aligned_features": aligned_features,
        "validated_features": validated_features,
        "profile": {
            "schema_source": profile["schema_source"],
            "target_column": profile["target_column"],
            "feature_count": profile["feature_count"],
        },
    }
    return result


def validate_batch(rows: list[dict[str, Any]], *, strict: bool = False) -> dict[str, Any]:
    # Run row-wise validation and aggregate the overall batch status.
    row_reports = []
    for index, row in enumerate(rows):
        report = validate_features(row, strict=strict)
        report["row_index"] = index
        row_reports.append(report)

    return {
        "count": len(rows),
        "strict": strict,
        "is_valid": all(report["is_valid"] for report in row_reports),
        "row_reports": row_reports,
    }


def build_sample_report() -> dict[str, Any]:
    # Validate the sample API payload so users can inspect a concrete report.
    if not SAMPLE_PAYLOAD_PATH.exists():
        return {
            "sample_payload_found": False,
            "message": f"Sample payload not found at {SAMPLE_PAYLOAD_PATH}",
        }

    with SAMPLE_PAYLOAD_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    features = payload.get("features", {})
    return {
        "sample_payload_found": True,
        "payload_path": str(SAMPLE_PAYLOAD_PATH.relative_to(PROJECT_ROOT)),
        "validation": validate_features(features, strict=False),
    }


def main() -> int:
    ensure_output_dirs()
    profile = load_validation_profile()
    save_json(PROFILE_PATH, profile)
    save_json(SAMPLE_REPORT_PATH, build_sample_report())

    print("Stage 07 validation profile generated.")
    print(f"Profile: {PROFILE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Sample report: {SAMPLE_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Feature count: {profile['feature_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())