"""FastAPI routes for the ML pipeline."""
import io
import logging
import os
import pickle
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sklearn.model_selection import train_test_split

from backend.data_processor.cleaner import (
    build_feature_pipeline,
    clean_dataframe,
    encode_target,
)
from backend.data_processor.profiler import (
    detect_problem_type,
    profile_dataset,
    suggest_preprocessing,
)
from backend.evaluator.evaluator import build_comparison_table
from backend.explainer.explainer import (
    generate_plain_english_explanation,
    get_feature_importance,
)
from backend.model_trainer.trainer import train_all_models

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
JOBS_DIR = Path(__file__).parent.parent / "jobs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job store (suitable for single-process deployment)
_job_store: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_feature_names(transformer) -> list[str]:
    """Extract feature names from a fitted ColumnTransformer."""
    names: list[str] = []
    for _, sub_pipe, cols in transformer.transformers_:
        if isinstance(cols, list):
            last_step = sub_pipe.steps[-1][1]
            if hasattr(last_step, "get_feature_names_out"):
                out_names = last_step.get_feature_names_out(cols)
                names.extend(out_names.tolist())
            else:
                names.extend(cols)
    return names


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a CSV file and return a job_id + data profile."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {exc}") from exc

    if df.empty or len(df.columns) < 2:
        raise HTTPException(
            status_code=400, detail="Dataset must have at least 2 columns and 1 row."
        )

    job_id = str(uuid.uuid4())

    # Persist raw CSV
    csv_path = UPLOAD_DIR / f"{job_id}.csv"
    csv_path.write_bytes(content)

    # Profile
    profile = profile_dataset(df)
    target_col = df.columns[-1]  # default suggestion
    problem_type = detect_problem_type(df, target_col)
    suggestions = suggest_preprocessing(profile, problem_type)

    _job_store[job_id] = {
        "status": "uploaded",
        "csv_path": str(csv_path),
        "columns": df.columns.tolist(),
        "profile": profile,
        "problem_type": problem_type,
        "target_col": target_col,
        "preprocessing_suggestions": suggestions,
    }

    logger.info("Job %s uploaded — %d rows, %d cols", job_id, *df.shape)

    return {
        "job_id": job_id,
        "columns": df.columns.tolist(),
        "profile": profile,
        "detected_problem_type": problem_type,
        "suggested_target": target_col,
        "preprocessing_suggestions": suggestions,
    }


@router.post("/run-pipeline/{job_id}")
async def run_pipeline(job_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """
    Trigger the full ML pipeline for an uploaded dataset.

    Body parameters (all optional):
    - target_col: column name to predict (default: last column)
    - tune: bool, whether to run Optuna tuning (default: true)
    """
    if job_id not in _job_store:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = _job_store[job_id]
    if job["status"] == "running":
        raise HTTPException(status_code=409, detail="Pipeline already running.")

    target_col: str = body.get("target_col", job["target_col"])
    tune: bool = body.get("tune", True)

    job["status"] = "running"

    try:
        # Load CSV
        df_raw = pd.read_csv(job["csv_path"])

        if target_col not in df_raw.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        # Re-profile with chosen target
        profile = profile_dataset(df_raw)
        problem_type = detect_problem_type(df_raw, target_col)
        suggestions = suggest_preprocessing(profile, problem_type)

        # Clean
        df_clean = clean_dataframe(df_raw, target_col)

        # Encode target
        y_raw = df_clean[target_col].dropna()
        df_clean = df_clean.loc[y_raw.index]
        y, target_encoder = encode_target(df_clean[target_col])

        # Build feature pipeline
        ct, feature_cols = build_feature_pipeline(df_clean, target_col)
        X = ct.fit_transform(df_clean[feature_cols])
        feature_names = _get_feature_names(ct)

        # Train/test split (stratified for classification)
        stratify = y if problem_type == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        # Train models
        trained_models, all_scores, best_model_name = train_all_models(
            X_train, y_train, X_test, y_test, problem_type, tune_best=tune
        )

        best_model = trained_models[best_model_name]
        best_scores = all_scores[best_model_name]

        # Comparison table
        comparison_table = build_comparison_table(all_scores, problem_type)

        # Feature importance
        feature_importances = get_feature_importance(
            best_model, feature_names, X_test, problem_type
        )

        # Plain-English explanation
        explanation = generate_plain_english_explanation(
            best_model_name,
            best_scores,
            feature_importances,
            problem_type,
            profile["shape"],
            suggestions,
        )

        # Persist model
        model_path = JOBS_DIR / f"{job_id}_model.pkl"
        joblib.dump({"model": best_model, "transformer": ct, "target_encoder": target_encoder}, model_path)

        result: dict[str, Any] = {
            "job_id": job_id,
            "problem_type": problem_type,
            "target_col": target_col,
            "best_model": best_model_name,
            "best_model_scores": best_scores,
            "comparison_table": comparison_table,
            "feature_importances": feature_importances,
            "explanation": explanation,
            "preprocessing_suggestions": suggestions,
            "dataset_shape": profile["shape"],
        }

        job.update({"status": "completed", "result": result})
        logger.info("Job %s completed. Best model: %s", job_id, best_model_name)
        return result

    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        logger.exception("Job %s failed: %s", job_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/results/{job_id}")
async def get_results(job_id: str) -> dict[str, Any]:
    """Fetch the results of a completed pipeline run."""
    if job_id not in _job_store:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = _job_store[job_id]
    if job["status"] == "running":
        return {"job_id": job_id, "status": "running"}
    if job["status"] == "failed":
        return {"job_id": job_id, "status": "failed", "error": job.get("error")}
    if job["status"] != "completed":
        return {"job_id": job_id, "status": job["status"]}

    return job["result"]


@router.get("/download-model/{job_id}")
async def download_model(job_id: str) -> FileResponse:
    """Download the trained best model as a .pkl file."""
    model_path = JOBS_DIR / f"{job_id}_model.pkl"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found. Run the pipeline first.")

    return FileResponse(
        path=str(model_path),
        media_type="application/octet-stream",
        filename=f"model_{job_id}.pkl",
    )
