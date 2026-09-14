from __future__ import annotations
"""
Build and apply reusable feature validation rules for inference payloads.

This module generates a data validation profile from the cleaned training dataset
and provides functions to validate incoming inference requests against this profile.
It ensures that the features provided to the model are numeric, finite, and
reasonably within the ranges observed during training.
"""

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# --- Project Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "01-eda" / "outputs" / "processed" / "usedcars_stage1.csv"
SAMPLE_PAYLOAD_PATH = PROJECT_ROOT / "06-inference-api" / "sample_predict_payload.json"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
PROFILE_PATH = METRICS_DIR / "data_validation_profile.json"
SAMPLE_REPORT_PATH = METRICS_DIR / "sample_validation_report.json"

# --- Constants ---
TARGET_COLUMN = "price"
# Ratio of missing features beyond which a sparse payload warning is triggered
MISSING_FEATURE_RATIO_WARNING = 0.35


def ensure_output_dirs() -> None:
    """Create necessary output directories for validation metrics if they don't exist."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def load_cleaned_data() -> pd.DataFrame:
    """
    Load the processed training dataset used to establish the validation baseline.

    Returns:
        pd.DataFrame: The cleaned training dataset.

    Raises:
        FileNotFoundError: If the processed dataset is not found at DATA_PATH.
        ValueError: If the expected target column is missing from the dataset.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' not found in cleaned dataset.")
    return df


def split_features(df: pd.DataFrame) -> list[str]:
    """
    Separate the feature columns from the target column in the dataframe.

    Args:
        df: The dataframe containing both features and target.

    Returns:
        A list of strings containing only the feature column names.
    """
    return [column for column in df.columns if column != TARGET_COLUMN]


def to_jsonable(value: Any) -> Any:
    """
    Convert NumPy types to standard Python types to ensure JSON serializability.

    Args:
        value: The value to convert (can be NumPy scalars, arrays, or Pandas objects).

    Returns:
        The value converted to a JSON-compatible Python type.
    """
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Series, pd.Index)):
        return value.tolist()
    return value


def numeric_summary(series: pd.Series) -> dict[str, Any]:
    """
    Calculate descriptive statistics for a numeric column to be used in validation.

    Args:
        series: The pandas Series to summarize.

    Returns:
        A dictionary containing min, max, median, mean, std, unique count, and binary flag.

    Raises:
        ValueError: If the column contains no valid numeric values.
    """
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
    """
    Generate and cache a validation profile based on the cleaned training data.

    The profile includes the expected features and their respective numeric summaries.
    Caching prevents re-calculating these statistics on every validation call.

    Returns:
        A dictionary containing the schema source, target column, feature list, and feature stats.
    """
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
    """
    Save a dictionary to a JSON file with indentation for readability.

    Args:
        path: The destination file path.
        payload: The dictionary to save.
    """
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _coerce_numeric(value: Any, feature_name: str) -> float:
    """
    Attempt to convert a value to a finite float.

    Args:
        value: The value to coerce.
        feature_name: The name of the feature (used for error reporting).

    Returns:
        The value as a float.

    Raises:
        ValueError: If the value cannot be converted to a float or is not finite (NaN/Inf).
    """
    # Enforce numeric finite values so downstream model math is safe.
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Feature '{feature_name}' must be numeric. Got: {value!r}") from exc

    if not math.isfinite(numeric_value):
        raise ValueError(f"Feature '{feature_name}' must be a finite number. Got: {value!r}")

    return numeric_value


def validate_features(features: dict[str, Any], *, strict: bool = False) -> dict[str, Any]:
    """
    Validate, coerce, and align incoming features to the training schema.

    This function checks for unknown features, missing features, and values that
    fall outside the ranges observed during training.

    Args:
        features: A dictionary of feature names and their corresponding values.
        strict: If True, unknown or missing features are treated as errors. If False,
                they are treated as warnings.

    Returns:
        A validation report dictionary containing validity status, feature counts,
        aligned features, and any errors or warnings encountered.
    """
    # Validate, coerce, and align incoming features to the full training schema.
    profile = load_validation_profile()
    expected_features = profile["features"]
    feature_stats = profile["feature_stats"]

    # Pre-fill aligned features with defaults (0.0) to ensure the model receives a complete vector
    aligned_features = {feature: 0.0 for feature in expected_features}
    unknown_features: list[str] = []
    missing_features: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    validated_features: dict[str, float] = {}

    for feature_name, value in features.items():
        # 1. Check for unknown features (fields not present in training data)
        if feature_name not in aligned_features:
            unknown_features.append(feature_name)
            message = f"Unknown feature '{feature_name}' ignored."
            if strict:
                errors.append(message)
            else:
                warnings.append(message)
            continue

        # 2. Coerce to numeric and ensure finiteness
        numeric_value = _coerce_numeric(value, feature_name)
        validated_features[feature_name] = numeric_value
        aligned_features[feature_name] = numeric_value

        # 3. Range validation: check if value is within [min, max] observed in training
        stat = feature_stats[feature_name]
        if numeric_value < stat["min"] or numeric_value > stat["max"]:
            warnings.append(
                f"Feature '{feature_name}' is outside the observed training range "
                f"[{stat['min']}, {stat['max']}]."
            )

        # 4. Binary validation: check if a binary feature received a non-binary value
        if stat["is_binary"] and numeric_value not in (0.0, 1.0):
            warnings.append(f"Feature '{feature_name}' is binary in training data but received {numeric_value}.")

    # 5. Check for missing features (expected fields not provided in payload)
    for feature_name in expected_features:
        if feature_name not in features:
            missing_features.append(feature_name)

    # 6. Evaluate missing feature severity
    missing_ratio = len(missing_features) / len(expected_features) if expected_features else 0.0
    if missing_features:
        message = (
            f"{len(missing_features)} expected features were not supplied and were defaulted to 0.0."
        )
        if strict:
            errors.append(message)
        else:
            warnings.append(message)

    # 7. Trigger sparse payload warning if too many features are missing
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
    """
    Run row-wise validation and aggregate the overall batch status.

    Args:
        rows: A list of feature dictionaries to validate.
        strict: Whether to treat unknown/missing features as errors.

    Returns:
        A dictionary containing the batch size, strict mode flag, overall validity,
        and individual row reports.
    """
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
    """
    Validate a sample API payload to provide a concrete example of a validation report.

    Returns:
        A dictionary containing whether the sample was found, the path to the sample,
        and the resulting validation report.
    """
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
    """
    Main execution entry point for generating the validation profile and sample report.

    Returns:
        0 on successful execution.
    """
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
