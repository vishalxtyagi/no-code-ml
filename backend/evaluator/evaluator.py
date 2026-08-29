"""Evaluation metrics for classification and regression."""
import numpy as np
from sklearn import metrics


def score_model(
    y_true: np.ndarray, y_pred: np.ndarray, problem_type: str
) -> dict[str, float]:
    """Return appropriate metrics given problem type."""
    if problem_type == "classification":
        y_pred_int = np.round(y_pred).astype(int)
        return {
            "accuracy": round(float(metrics.accuracy_score(y_true, y_pred_int)), 4),
            "f1_weighted": round(
                float(metrics.f1_score(y_true, y_pred_int, average="weighted", zero_division=0)),
                4,
            ),
            "precision": round(
                float(metrics.precision_score(y_true, y_pred_int, average="weighted", zero_division=0)),
                4,
            ),
            "recall": round(
                float(metrics.recall_score(y_true, y_pred_int, average="weighted", zero_division=0)),
                4,
            ),
        }
    else:
        return {
            "mae": round(float(metrics.mean_absolute_error(y_true, y_pred)), 4),
            "mse": round(float(metrics.mean_squared_error(y_true, y_pred)), 4),
            "rmse": round(float(np.sqrt(metrics.mean_squared_error(y_true, y_pred))), 4),
            "r2": round(float(metrics.r2_score(y_true, y_pred)), 4),
        }


def build_comparison_table(
    scores: dict[str, dict[str, float]], problem_type: str
) -> list[dict]:
    """Return a list of dicts suitable for JSON serialisation."""
    rows = []
    for model_name, model_scores in scores.items():
        row = {"model": model_name}
        row.update(model_scores)
        rows.append(row)

    # Sort descending by primary metric
    if problem_type == "classification":
        rows.sort(key=lambda r: r.get("f1_weighted", 0), reverse=True)
    else:
        rows.sort(key=lambda r: r.get("rmse", float("inf")))

    return rows
