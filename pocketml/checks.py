"""Validity checks that run before any score is shown: "is this model lying to you?"."""
import warnings as _pywarnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

LEAK_THRESHOLD = 0.98  # single-column CV score (balanced acc / R²) that counts as "too good"
MIN_ROWS, MIN_PER_CLASS, MIN_CLASS_SHARE = 50, 10, 0.05


def _warn(level, code, message, column=None):
    return {"level": level, "code": code, "column": column, "message": message}


def _looks_datetime(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    sample = s.dropna().astype(str).head(200)
    if sample.empty or sample.str.len().mean() > 40:
        return False
    with _pywarnings.catch_warnings():
        _pywarnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.9


def _single_column_score(x, y, task):
    """CV score of a shallow tree that sees only this one column."""
    if pd.api.types.is_numeric_dtype(x):
        X = x.astype(float).fillna(-1e9).to_frame()
    else:
        X = x.astype("category").cat.codes.to_frame()
    if task == "classification":
        folds = min(3, int(y.value_counts().min()))
        if folds < 2:
            return None
        model = DecisionTreeClassifier(max_depth=3, random_state=0)
        cv, scoring = StratifiedKFold(folds, shuffle=True, random_state=0), "balanced_accuracy"
    else:
        model = DecisionTreeRegressor(max_depth=3, random_state=0)
        cv, scoring = KFold(3, shuffle=True, random_state=0), "r2"
    return float(np.mean(cross_val_score(model, X, y, cv=cv, scoring=scoring)))


def run_checks(df, target, task, kinds, keep=()):
    """Return (warnings, excluded_columns). `df` must already have target NaNs dropped."""
    out, excluded = [], []
    n, y = len(df), df[target]
    features = [c for c in df.columns if c != target]

    for c in features:
        s = df[c]
        if c in keep or kinds[c] == "text":
            continue
        if _looks_datetime(s):
            excluded.append(c)
            out.append(_warn("excluded", "datetime", f"'{c}' looks like a date. Excluded: random CV on time-ordered "
                             "data overstates accuracy; time-aware validation is not supported yet.", c))
            continue
        integral = pd.api.types.is_integer_dtype(s) or not pd.api.types.is_numeric_dtype(s)
        if integral and n >= 20 and s.nunique(dropna=True) == n:
            excluded.append(c)
            out.append(_warn("excluded", "id_like", f"'{c}' is unique on every row (an ID). Excluded: "
                             "models memorise IDs instead of learning.", c))
            continue
        if task == "classification" and not pd.api.types.is_numeric_dtype(s) and y.nunique() > 1:
            groups = df.groupby(c, dropna=True, observed=True)[target].nunique()
            if 1 < len(groups) <= n / 2 and (groups == 1).all():
                excluded.append(c)
                out.append(_warn("excluded", "leak_mapping", f"'{c}' maps one-to-one onto the target. Likely "
                                 "leakage or a relabelled target. Excluded (use --keep to override).", c))
                continue
        score = _single_column_score(s, y, task)
        if score is not None and score >= LEAK_THRESHOLD:
            excluded.append(c)
            out.append(_warn("excluded", "leak_single_column", f"'{c}' alone predicts the target almost perfectly "
                             f"(CV {score:.3f}). Likely leakage or a proxy. Excluded (use --keep to override).", c))

    if n < MIN_ROWS:
        out.append(_warn("warning", "too_small", f"Only {n} rows; scores will be noisy. Aim for {MIN_ROWS}+."))
    if task == "classification":
        counts = y.value_counts()
        if counts.min() < MIN_PER_CLASS:
            out.append(_warn("warning", "too_small_class", f"Class '{counts.idxmin()}' has only {counts.min()} rows "
                             f"(< {MIN_PER_CLASS}); its scores are unreliable."))
        share = counts.min() / n
        if share < MIN_CLASS_SHARE:
            out.append(_warn("warning", "imbalance", f"Minority class '{counts.idxmin()}' is {share:.1%} of rows. "
                             "Accuracy would mislead; balanced metrics are used."))

    usable = [c for c in features if c not in excluded]
    if usable:
        dup_features = int(df.duplicated(subset=usable).sum())
        if dup_features:
            per_group = df.groupby(usable, dropna=False, observed=True)[target].nunique()
            conflicts = int((per_group > 1).sum())
            msg = f"{dup_features} rows duplicate another row's features; CV may score them on rows it trained on."
            if conflicts:
                msg += f" {conflicts} of those groups have conflicting targets (label noise)."
            out.append(_warn("warning", "duplicates", msg))
    return out, excluded
