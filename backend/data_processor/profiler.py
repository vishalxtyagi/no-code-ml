"""Data profiling and problem-type detection."""
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Generate a comprehensive profile of the dataset."""
    profile: dict[str, Any] = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": {},
        "missing_summary": {},
        "duplicate_rows": int(df.duplicated().sum()),
    }

    for col in df.columns:
        series = df[col]
        col_info: dict[str, Any] = {
            "dtype": str(series.dtype),
            "missing_count": int(series.isnull().sum()),
            "missing_pct": round(series.isnull().mean() * 100, 2),
            "unique_count": int(series.nunique()),
        }

        if pd.api.types.is_numeric_dtype(series):
            col_info["kind"] = "numerical"
            desc = series.describe()
            col_info["stats"] = {
                k: round(float(v), 4)
                for k, v in desc.items()
                if k != "count"
            }
            # Outlier detection via IQR
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            outlier_mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
            col_info["outlier_count"] = int(outlier_mask.sum())
        else:
            col_info["kind"] = "categorical"
            top_values = series.value_counts().head(5).to_dict()
            col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}

        profile["columns"][col] = col_info

    profile["missing_summary"] = {
        col: info["missing_pct"]
        for col, info in profile["columns"].items()
        if info["missing_pct"] > 0
    }

    return profile


def detect_problem_type(df: pd.DataFrame, target_col: str) -> str:
    """Detect whether the problem is classification or regression."""
    target = df[target_col].dropna()

    if pd.api.types.is_object_dtype(target) or pd.api.types.is_bool_dtype(target):
        return "classification"

    unique_ratio = target.nunique() / len(target)
    if target.nunique() <= 20 or unique_ratio < 0.05:
        return "classification"

    return "regression"


def select_target_column(df: pd.DataFrame) -> str:
    """Heuristically suggest the target column (last column by default)."""
    return df.columns[-1]


def suggest_preprocessing(profile: dict[str, Any], problem_type: str) -> list[str]:
    """Return human-readable preprocessing suggestions."""
    suggestions: list[str] = []

    missing_cols = [
        col
        for col, info in profile["columns"].items()
        if info["missing_pct"] > 0
    ]
    if missing_cols:
        suggestions.append(
            f"Impute missing values in columns: {', '.join(missing_cols)}"
        )

    categorical_cols = [
        col
        for col, info in profile["columns"].items()
        if info["kind"] == "categorical"
    ]
    if categorical_cols:
        suggestions.append(
            f"Encode categorical columns: {', '.join(categorical_cols)}"
        )

    numerical_cols = [
        col
        for col, info in profile["columns"].items()
        if info["kind"] == "numerical"
    ]
    if numerical_cols:
        suggestions.append(
            f"Scale numerical features: {', '.join(numerical_cols)}"
        )

    outlier_cols = [
        col
        for col, info in profile["columns"].items()
        if info.get("outlier_count", 0) > 0
    ]
    if outlier_cols:
        suggestions.append(
            f"Outliers detected in: {', '.join(outlier_cols)} — consider clipping or robust scaling"
        )

    if problem_type == "classification":
        suggestions.append(
            "Problem type: Classification — will use accuracy and F1-score metrics"
        )
    else:
        suggestions.append(
            "Problem type: Regression — will use RMSE and MAE metrics"
        )

    return suggestions
