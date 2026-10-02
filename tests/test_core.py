import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from nocodeml import core


def _clf_df(n=300, seed=0, weights=None):
    X, y = make_classification(n_samples=n, n_features=6, n_informative=4, random_state=seed, weights=weights)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["label"] = np.where(y == 1, "yes", "no")
    return df


def _codes(report):
    return {w["code"] for w in report["warnings"]}


def _score(report, tier, metric):
    return next(c for c in report["candidates"] if c["tier"] == tier)["metrics"][metric]["mean"]


def test_load_normalises_placeholders(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("a,b,c\n1, x ,?\n2,N/A,3.5\n3,y,4\n")
    df = core.load(p)
    assert pd.api.types.is_numeric_dtype(df["c"]) and df["c"].isna().sum() == 1
    assert df["b"].isna().sum() == 1 and df["b"].iloc[0] == "x"


def test_detect_task():
    assert core.detect_task(pd.Series(["a", "b"])) == "classification"
    assert core.detect_task(pd.Series([0, 1, 1, 0])) == "classification"
    assert core.detect_task(pd.Series(np.linspace(0, 1, 100))) == "regression"


def test_leaks_and_ids_are_excluded():
    df = _clf_df()
    df["row_id"] = range(len(df))
    df["status_code"] = df["label"].map({"yes": "S1", "no": "S0"})  # relabelled target
    df["leaky_num"] = (df["label"] == "yes") * 10.0 + np.random.default_rng(0).normal(0, 0.01, len(df))
    report, _ = core.evaluate(df, "label")
    assert {"row_id", "status_code", "leaky_num"} <= set(report["excluded"])
    assert {"id_like", "leak_mapping", "leak_single_column"} <= _codes(report)
    assert not {"f0", "f1", "f2"} & set(report["excluded"])


def test_keep_overrides_exclusion():
    df = _clf_df()
    df["status_code"] = df["label"].map({"yes": "S1", "no": "S0"})
    report, _ = core.evaluate(df, "label", keep=["status_code"])
    assert "status_code" not in report["excluded"]


def test_imbalance_small_and_duplicates_flagged():
    df = _clf_df(n=400, weights=[0.97])
    assert "imbalance" in _codes(core.evaluate(df, "label")[0])
    small = _clf_df(n=40)
    assert "too_small" in _codes(core.evaluate(small, "label")[0])
    dup = _clf_df()
    extra = dup.iloc[:5].copy()
    extra["label"] = np.where(extra["label"] == "yes", "no", "yes")
    report, _ = core.evaluate(pd.concat([dup, extra], ignore_index=True), "label")
    assert any(w["code"] == "duplicates" and "conflicting" in w["message"] for w in report["warnings"])


def test_missing_target_rows_dropped():
    df = _clf_df()
    df.loc[:9, "label"] = np.nan
    report, _ = core.evaluate(df, "label")
    assert report["n_rows"] == 290 and "missing_target" in _codes(report)


def test_classification_and_regression_beat_floor():
    report, bundle = core.evaluate(_clf_df(), "label")
    assert _score(report, 2, "balanced_accuracy") > _score(report, 0, "balanced_accuracy") + 0.2
    assert set(bundle["model"].predict(_clf_df().drop(columns="label"))) <= {"yes", "no"}
    X, y = make_regression(n_samples=300, n_features=5, noise=5, random_state=0)
    df = pd.DataFrame(X, columns=list("abcde")).assign(price=y)
    report, _ = core.evaluate(df, "price")
    assert report["task"] == "regression" and _score(report, 2, "r2") > 0.8


def test_text_classification():
    rng = np.random.default_rng(0)
    good, bad = ["great", "love", "excellent", "happy"], ["awful", "broken", "refund", "angry"]
    filler = "the order arrived on tuesday and I opened the box to check it".split()
    rows = []
    for i in range(240):
        pos = i % 2 == 0
        words = list(rng.choice(filler, 10)) + list(rng.choice(good if pos else bad, 2))
        rng.shuffle(words)
        rows.append({"review": " ".join(words), "sentiment": "pos" if pos else "neg"})
    report, _ = core.evaluate(pd.DataFrame(rows), "sentiment")
    tiers = {c["tier"]: c for c in report["candidates"]}
    assert tiers[1]["status"] == "skipped" and "milestone 2" in tiers[1]["skip_reason"]
    assert report["winner"]["tier"] == 2 and _score(report, 2, "balanced_accuracy") > 0.9


def test_zero_training_tier_reports_skip_or_runs():
    report, _ = core.evaluate(_clf_df(n=150), "label")
    tier1 = next(c for c in report["candidates"] if c["tier"] == 1)
    try:
        import tabicl  # noqa: F401
    except ImportError:
        assert tier1["status"] == "skipped" and "nocodeml[fm]" in tier1["skip_reason"]
    else:
        assert tier1["status"] == "ok", tier1["skip_reason"]


def _cli(*args, cwd):
    return subprocess.run([sys.executable, "-m", "nocodeml.cli", *args], cwd=cwd, capture_output=True, text=True)


def test_cli_train_predict_and_concurrency(tmp_path):
    _clf_df(seed=1).to_csv(tmp_path / "a.csv", index=False)
    X, y = make_regression(n_samples=200, n_features=4, random_state=2)
    pd.DataFrame(X, columns=list("wxyz")).assign(t=y).to_csv(tmp_path / "b.csv", index=False)

    procs = [subprocess.Popen([sys.executable, "-m", "nocodeml.cli", "train", f, "--target", t, "--out", "runs"],
                              cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for f, t in (("a.csv", "label"), ("b.csv", "t"))]
    for pr in procs:
        out, err = pr.communicate(timeout=600)
        assert pr.returncode == 0, err
    runs = sorted((tmp_path / "runs").iterdir())
    assert len(runs) == 2
    tasks = {json.loads((r / "report.json").read_text())["task"] for r in runs}
    assert tasks == {"classification", "regression"}
    for r in runs:
        assert {"report.json", "report.md", "model.joblib"} <= {p.name for p in r.iterdir()}

    clf_run = next(r for r in runs if json.loads((r / "report.json").read_text())["target"] == "label")
    res = _cli("predict", str(clf_run), "a.csv", "--out", "preds.csv", cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    preds = pd.read_csv(tmp_path / "preds.csv")
    assert len(preds) == 300 and "predicted_label" in preds


def test_cli_missing_target_lists_columns(tmp_path):
    _clf_df().to_csv(tmp_path / "a.csv", index=False)
    res = _cli("train", "a.csv", cwd=tmp_path)
    assert res.returncode != 0 and "f0" in res.stderr
