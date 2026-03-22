"""Integration tests for the AutoML backend pipeline."""
import io
import sys
import os

# Ensure the project root is on PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.data_processor.cleaner import clean_dataframe, encode_target, build_feature_pipeline
from backend.data_processor.profiler import detect_problem_type, profile_dataset
from backend.evaluator.evaluator import score_model, build_comparison_table


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classification_csv() -> bytes:
    """Simple binary classification CSV."""
    df = pd.DataFrame({
        "age": [25, 32, 45, 23, 52, 41, 30, 28, 60, 35],
        "income": [40000, 55000, 80000, 35000, 95000, 72000, 48000, 43000, 110000, 60000],
        "education": ["bachelor", "master", "phd", "bachelor", "phd", "master", "bachelor", "bachelor", "phd", "master"],
        "label": ["no", "no", "yes", "no", "yes", "yes", "no", "no", "yes", "yes"],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture
def regression_csv() -> bytes:
    """Simple regression CSV."""
    rng = np.random.default_rng(0)
    n = 60
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 5, n)
    y = 3 * x1 + 2 * x2 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "target": y})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture
def api_client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Unit tests: profiler
# ---------------------------------------------------------------------------

class TestProfiler:
    def test_profile_shape(self, classification_csv):
        df = pd.read_csv(io.BytesIO(classification_csv))
        profile = profile_dataset(df)
        assert profile["shape"]["rows"] == 10
        assert profile["shape"]["columns"] == 4

    def test_detect_classification(self, classification_csv):
        df = pd.read_csv(io.BytesIO(classification_csv))
        assert detect_problem_type(df, "label") == "classification"

    def test_detect_regression(self, regression_csv):
        df = pd.read_csv(io.BytesIO(regression_csv))
        assert detect_problem_type(df, "target") == "regression"

    def test_profile_missing_summary(self):
        df = pd.DataFrame({
            "a": [1, None, 3, 4, 5],
            "b": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0],
        })
        profile = profile_dataset(df)
        assert "a" in profile["missing_summary"]
        assert profile["columns"]["a"]["missing_pct"] == 20.0


# ---------------------------------------------------------------------------
# Unit tests: cleaner
# ---------------------------------------------------------------------------

class TestCleaner:
    def test_clean_removes_nulls_and_encodes(self, classification_csv):
        df = pd.read_csv(io.BytesIO(classification_csv))
        cleaned = clean_dataframe(df, "label")
        assert cleaned.isnull().sum().sum() == 0

    def test_encode_target_classification(self, classification_csv):
        df = pd.read_csv(io.BytesIO(classification_csv))
        encoded, le = encode_target(df["label"])
        assert le is not None
        assert set(encoded).issubset({0, 1})

    def test_encode_target_regression(self, regression_csv):
        df = pd.read_csv(io.BytesIO(regression_csv))
        encoded, le = encode_target(df["target"])
        assert le is None
        assert encoded.dtype == float

    def test_feature_pipeline_returns_array(self, classification_csv):
        df = pd.read_csv(io.BytesIO(classification_csv))
        df = clean_dataframe(df, "label")
        ct, _ = build_feature_pipeline(df, "label")
        X = ct.fit_transform(df.drop(columns=["label"]))
        assert X.ndim == 2
        assert X.shape[0] == len(df)


# ---------------------------------------------------------------------------
# Unit tests: evaluator
# ---------------------------------------------------------------------------

class TestEvaluator:
    def test_classification_scores(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        scores = score_model(y_true, y_pred, "classification")
        assert "accuracy" in scores
        assert "f1_weighted" in scores
        assert 0 <= scores["accuracy"] <= 1

    def test_regression_scores(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9])
        scores = score_model(y_true, y_pred, "regression")
        assert "rmse" in scores
        assert "mae" in scores
        assert "r2" in scores
        assert scores["rmse"] >= 0

    def test_comparison_table_sorted_classification(self):
        scores = {
            "ModelA": {"f1_weighted": 0.9, "accuracy": 0.92},
            "ModelB": {"f1_weighted": 0.75, "accuracy": 0.78},
        }
        table = build_comparison_table(scores, "classification")
        assert table[0]["model"] == "ModelA"

    def test_comparison_table_sorted_regression(self):
        scores = {
            "ModelA": {"rmse": 1.5, "mae": 1.0},
            "ModelB": {"rmse": 0.5, "mae": 0.3},
        }
        table = build_comparison_table(scores, "regression")
        assert table[0]["model"] == "ModelB"


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------

class TestAPI:
    def test_upload_csv(self, api_client, classification_csv):
        response = api_client.post(
            "/api/upload",
            files={"file": ("test.csv", classification_csv, "text/csv")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert "profile" in data
        assert data["detected_problem_type"] == "classification"

    def test_upload_invalid_file(self, api_client):
        response = api_client.post(
            "/api/upload",
            files={"file": ("test.txt", b"not a csv", "text/plain")},
        )
        assert response.status_code == 400

    def test_full_pipeline_classification(self, api_client, classification_csv):
        upload_resp = api_client.post(
            "/api/upload",
            files={"file": ("test.csv", classification_csv, "text/csv")},
        )
        assert upload_resp.status_code == 200
        job_id = upload_resp.json()["job_id"]

        run_resp = api_client.post(
            f"/api/run-pipeline/{job_id}",
            json={"target_col": "label", "tune": False},
        )
        assert run_resp.status_code == 200
        result = run_resp.json()

        assert result["problem_type"] == "classification"
        assert "best_model" in result
        assert "best_model_scores" in result
        assert "comparison_table" in result
        assert len(result["comparison_table"]) > 0
        assert "feature_importances" in result
        assert "explanation" in result
        assert len(result["explanation"]) > 50

    def test_full_pipeline_regression(self, api_client, regression_csv):
        upload_resp = api_client.post(
            "/api/upload",
            files={"file": ("test.csv", regression_csv, "text/csv")},
        )
        assert upload_resp.status_code == 200
        job_id = upload_resp.json()["job_id"]

        run_resp = api_client.post(
            f"/api/run-pipeline/{job_id}",
            json={"target_col": "target", "tune": False},
        )
        assert run_resp.status_code == 200
        result = run_resp.json()

        assert result["problem_type"] == "regression"
        assert "rmse" in result["best_model_scores"]

    def test_get_results(self, api_client, classification_csv):
        upload_resp = api_client.post(
            "/api/upload",
            files={"file": ("test.csv", classification_csv, "text/csv")},
        )
        job_id = upload_resp.json()["job_id"]
        api_client.post(
            f"/api/run-pipeline/{job_id}",
            json={"target_col": "label", "tune": False},
        )
        results_resp = api_client.get(f"/api/results/{job_id}")
        assert results_resp.status_code == 200
        assert results_resp.json()["best_model"] is not None

    def test_results_not_found(self, api_client):
        response = api_client.get("/api/results/nonexistent-job-id")
        assert response.status_code == 404

    def test_download_model(self, api_client, classification_csv):
        upload_resp = api_client.post(
            "/api/upload",
            files={"file": ("test.csv", classification_csv, "text/csv")},
        )
        job_id = upload_resp.json()["job_id"]
        api_client.post(
            f"/api/run-pipeline/{job_id}",
            json={"target_col": "label", "tune": False},
        )
        dl_resp = api_client.get(f"/api/download-model/{job_id}")
        assert dl_resp.status_code == 200
        assert dl_resp.headers["content-type"] == "application/octet-stream"
