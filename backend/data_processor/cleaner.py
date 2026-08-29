"""Data cleaning, imputation, encoding, and scaling pipeline."""
import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

_MAX_CATEGORICAL_UNIQUE = 50  # drop high-cardinality columns after target extraction


def clean_dataframe(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Perform light cleaning on the raw DataFrame:
    - Strip whitespace from string columns
    - Replace common null placeholders
    - Drop columns with >50 % missing values
    - Drop constant columns (zero variance)
    """
    df = df.copy()

    # Normalise null placeholders
    null_placeholders = {"?", "NA", "N/A", "na", "n/a", "null", "NULL", "none", "None", ""}
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].str.strip()
            df[col] = df[col].replace(list(null_placeholders), np.nan)

    # Drop columns with >50 % missing (except target)
    missing_pct = df.drop(columns=[target_col]).isnull().mean()
    drop_cols = missing_pct[missing_pct > 0.5].index.tolist()
    if drop_cols:
        logger.info("Dropping high-missing columns: %s", drop_cols)
        df = df.drop(columns=drop_cols)

    # Drop constant columns (except target)
    feature_cols = [c for c in df.columns if c != target_col]
    constant_cols = [c for c in feature_cols if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        logger.info("Dropping constant columns: %s", constant_cols)
        df = df.drop(columns=constant_cols)

    # Drop duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        logger.info("Dropped %d duplicate rows", before - len(df))

    return df


def encode_target(series: pd.Series) -> tuple[np.ndarray, LabelEncoder | None]:
    """Encode target column; returns array and encoder (or None for numeric)."""
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(dtype=float), None
    le = LabelEncoder()
    encoded = le.fit_transform(series.astype(str))
    return encoded, le


def build_feature_pipeline(
    df: pd.DataFrame, target_col: str
) -> tuple[ColumnTransformer, list[str]]:
    """
    Build a sklearn ColumnTransformer for features.
    Returns (transformer, feature_column_names_before_transform).
    """
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]

    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Remove very high cardinality categorical cols to avoid memory blowup
    categorical_cols = [
        c for c in categorical_cols if X[c].nunique() <= _MAX_CATEGORICAL_UNIQUE
    ]

    transformers = []
    if numerical_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, numerical_cols))

    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("No usable feature columns after cleaning.")

    ct = ColumnTransformer(transformers=transformers, remainder="drop")
    return ct, feature_cols
