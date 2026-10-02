"""Load data, compare model tiers under cross-validation, pick a winner, save the run."""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from nocodeml.checks import run_checks

NULLS = ["?", "NA", "N/A", "na", "n/a", "null", "NULL", "none", "None", ""]
METRICS = {
    "classification": {"balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro"},
    "regression": {"r2": "r2", "mae": "neg_mean_absolute_error"},
}
PRIMARY = {"classification": "balanced_accuracy", "regression": "r2"}


def load(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, keep_default_na=False, na_values=NULLS)
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        s = s.astype(object).where(s.notna(), None).map(lambda v: v.strip() if isinstance(v, str) else v)
        s = s.where(~s.isin(NULLS) & s.notna(), np.nan)
        # columns that only looked non-numeric because of placeholders like "?" become numeric again
        numeric = pd.to_numeric(s, errors="coerce")
        df[c] = numeric if s.notna().any() and numeric.notna().sum() == s.notna().sum() else s
    return df


def detect_task(y):
    y = y.dropna()
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"
    if y.nunique() <= 20 and np.allclose(y, np.round(y)):
        return "classification"
    return "regression"


def column_kinds(df):
    kinds = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            kinds[c] = "numeric"
        elif s.dropna().astype(str).str.len().mean() > 30 and s.nunique() > 0.5 * s.notna().sum():
            kinds[c] = "text"
        else:
            kinds[c] = "categorical"
    return kinds


def _as_text(x):
    return pd.Series(np.asarray(x).ravel()).fillna("").astype(str)


def _as_str(x):
    return pd.DataFrame(x).astype(object).where(pd.DataFrame(x).notna(), "<missing>").astype(str)


def _tabular_pipeline(num, cat, task):
    """Tier 2 for tables: gradient boosting with sklearn defaults (handles NaN natively)."""
    ct = ColumnTransformer(
        [("num", "passthrough", num),
         ("cat", make_pipeline(FunctionTransformer(_as_str), OrdinalEncoder(
             handle_unknown="use_encoded_value", unknown_value=-1, max_categories=250)), cat)],
        sparse_threshold=0)
    mask = [False] * len(num) + [True] * len(cat)
    est = (HistGradientBoostingClassifier if task == "classification" else HistGradientBoostingRegressor)(
        categorical_features=mask if any(mask) else None, random_state=0)
    return Pipeline([("prep", ct), ("model", est)])


def _text_pipeline(num, cat, text, task):
    """Tier 2 when text columns exist: TF-IDF + linear model over all columns."""
    parts = [(f"text_{c}", make_pipeline(FunctionTransformer(_as_text), TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, max_features=50_000)), c) for c in text]
    if num:
        parts.append(("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num))
    if cat:
        parts.append(("cat", make_pipeline(FunctionTransformer(_as_str), OneHotEncoder(handle_unknown="ignore")),
                      cat))
    est = LogisticRegression(max_iter=2000, class_weight="balanced") if task == "classification" else Ridge()
    return Pipeline([("prep", ColumnTransformer(parts)), ("model", est)])


def _zero_training(num, cat, task):
    """Tier 1: a tabular foundation model that predicts in-context, no gradient training. Optional extra."""
    try:
        from tabicl import TabICLClassifier, TabICLRegressor
    except ImportError:
        return None
    ct = ColumnTransformer(
        [("num", "passthrough", num),
         ("cat", make_pipeline(FunctionTransformer(_as_str), OrdinalEncoder(
             handle_unknown="use_encoded_value", unknown_value=-1)), cat)],
        sparse_threshold=0)
    est = TabICLClassifier() if task == "classification" else TabICLRegressor()
    return Pipeline([("prep", ct), ("model", est)])


def build_candidates(kinds, features, task):
    num = [c for c in features if kinds[c] == "numeric"]
    cat = [c for c in features if kinds[c] == "categorical"]
    text = [c for c in features if kinds[c] == "text"]
    floor = DummyClassifier(strategy="prior") if task == "classification" else DummyRegressor()
    cands = [{"tier": 0, "name": "floor (always predict the most common / mean value)", "model": floor}]
    if num or cat:
        fm = _zero_training(num, cat, task)
        note = " (non-text columns only)" if text else ""
        cands.append({"tier": 1, "name": f"zero-training foundation model (TabICL){note}", "model": fm,
                      "skip_reason": None if fm else "not installed: pip install 'nocodeml[fm]'"})
    else:
        cands.append({"tier": 1, "name": "zero-training model", "model": None,
                      "skip_reason": "text-only data: zero-training text models arrive in milestone 2"})
    tier2 = _text_pipeline(num, cat, text, task) if text else _tabular_pipeline(num, cat, task)
    cands.append({"tier": 2, "name": "TF-IDF + linear model" if text else "gradient boosting (defaults)",
                  "model": tier2})
    return cands


def _cv(task, y):
    if task == "classification":
        folds = min(5, int(y.value_counts().min()))
        if folds < 2:
            raise ValueError("Every class needs at least 2 rows to cross-validate. "
                             f"Smallest class: '{y.value_counts().idxmin()}'.")
        return StratifiedKFold(folds, shuffle=True, random_state=0)
    if len(y) < 10:
        raise ValueError(f"Need at least 10 rows for regression, got {len(y)}.")
    return KFold(5, shuffle=True, random_state=0)


def _pick(cands, task):
    """Tier 2 must beat tier 1 by more than one CV std to win; otherwise keep the no-training model."""
    ok = {c["tier"]: c for c in cands if c.get("status") == "ok"}
    floor, zero, trained = ok.get(0), ok.get(1), ok.get(2)
    m = PRIMARY[task]

    def score(c):
        return c["metrics"][m]["mean"]

    if zero and trained:
        gap = score(trained) - score(zero)
        if gap > trained["metrics"][m]["std"]:
            best, why = trained, f"trained model beat the zero-training model by {gap:.4f}, more than its CV std"
        else:
            best, why = zero, "trained model did not clearly beat the zero-training model, so no training is needed"
    else:
        best, why = zero or trained or floor, "best available tier"
    if best is not floor and floor and score(best) <= score(floor):
        why += "; WARNING: it does not beat the floor, the features may carry no signal"
    return best, why


def evaluate(df, target, keep=()):
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found. Columns: {', '.join(map(str, df.columns))}")
    n_missing = int(df[target].isna().sum())
    df = df[df[target].notna()].reset_index(drop=True)
    task = detect_task(df[target])
    kinds = column_kinds(df.drop(columns=[target]))
    warnings, excluded = run_checks(df, target, task, kinds, keep=keep)
    if n_missing:
        warnings.insert(0, {"level": "warning", "code": "missing_target", "column": target,
                            "message": f"{n_missing} rows had no target value and were dropped."})
    features = [c for c in kinds if c not in excluded]
    if not features:
        raise ValueError("No usable feature columns left after checks.")
    X, y = df[features], df[target]
    if task == "classification" and not pd.api.types.is_numeric_dtype(y):
        y = y.astype(str)
    cv = _cv(task, y)

    cands = build_candidates(kinds, features, task)
    for c in cands:
        if c["model"] is None:
            c["status"] = "skipped"
            continue
        try:
            res = cross_validate(c["model"], X, y, cv=cv, scoring=METRICS[task], error_score="raise")
        except Exception as exc:  # one failing tier must not sink the run; it is reported instead
            c["status"], c["skip_reason"] = "failed", f"{type(exc).__name__}: {exc}"
            continue
        c["status"], c["metrics"] = "ok", {}
        for k in METRICS[task]:
            vals = res[f"test_{k}"] * (-1 if k == "mae" else 1)
            c["metrics"][k] = {"mean": round(float(vals.mean()), 4), "std": round(float(vals.std()), 4)}

    best, why = _pick(cands, task)
    model = best["model"].fit(X, y)
    report = {
        "task": task, "target": target, "n_rows": len(df), "features": features, "excluded": excluded,
        "warnings": warnings,
        "candidates": [{k: c.get(k) for k in ("tier", "name", "status", "metrics", "skip_reason")} for c in cands],
        "winner": {"tier": best["tier"], "name": best["name"]}, "reason": why,
    }
    return report, {"model": model, "features": features, "target": target, "task": task}


def to_markdown(r):
    lines = [f"# Run report: predict `{r['target']}` ({r['task']}, {r['n_rows']} rows)", "", "## Validity checks"]
    lines += [f"- **{w['level']}** `{w['code']}`: {w['message']}" for w in r["warnings"]] or ["- No issues found."]
    keys = list(METRICS[r["task"]])
    lines += ["", "## Models (cross-validation, mean ± std)", "",
              "| Tier | Model | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for c in r["candidates"]:
        if c["status"] == "ok":
            cells = [f"{c['metrics'][k]['mean']:.4f} ± {c['metrics'][k]['std']:.4f}" for k in keys]
        else:
            cells = [f"{c['status']}: {c['skip_reason']}"] + [""] * (len(keys) - 1)
        lines.append(f"| {c['tier']} | {c['name']} | " + " | ".join(cells) + " |")
    lines += ["", f"**Winner:** tier {r['winner']['tier']}, {r['winner']['name']}. "
              f"Selected on `{PRIMARY[r['task']]}`: {r['reason']}.",
              "", f"Features used: {', '.join(map(str, r['features']))}"]
    return "\n".join(lines) + "\n"


def save(out_dir, report, bundle):
    run = Path(out_dir) / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    run.mkdir(parents=True)  # exist_ok=False: concurrent runs can never share a directory
    (run / "report.json").write_text(json.dumps(report, indent=2, default=str))
    (run / "report.md").write_text(to_markdown(report))
    joblib.dump(bundle, run / "model.joblib")
    return run


def predict(run_dir, df):
    bundle = joblib.load(Path(run_dir) / "model.joblib")  # pickle: only load runs you created yourself
    out = df.copy()
    out[f"predicted_{bundle['target']}"] = bundle["model"].predict(df.reindex(columns=bundle["features"]))
    return out
