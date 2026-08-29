"""Feature importance and plain-English explanation generation."""
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_feature_importance(
    model: Any,
    feature_names: list[str],
    X_sample: np.ndarray,
    problem_type: str,
    top_n: int = 15,
) -> list[dict[str, Any]]:
    """
    Return top-N feature importances.
    Falls back to SHAP if tree-based, otherwise uses built-in feature_importances_
    or permutation importance.
    """
    importances: np.ndarray | None = None

    # 1. Built-in feature_importances_ (tree-based models)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        logger.info("Using model.feature_importances_")

    # 2. Coefficient magnitude (linear models)
    elif hasattr(model, "coef_"):
        coef = np.abs(model.coef_)
        if coef.ndim > 1:
            coef = coef.mean(axis=0)
        importances = coef
        logger.info("Using model.coef_ for feature importance")

    # 3. SHAP fallback for tree-based
    else:
        try:
            import shap
            explainer = shap.Explainer(model, X_sample[:100])
            shap_values = explainer(X_sample[:100])
            vals = np.abs(shap_values.values)
            if vals.ndim == 3:
                vals = vals.mean(axis=2)
            importances = vals.mean(axis=0)
            logger.info("Using SHAP values for feature importance")
        except Exception as exc:
            logger.warning("SHAP failed: %s — returning empty importance", exc)
            return []

    if importances is None or len(importances) == 0:
        return []

    # Normalise
    total = importances.sum()
    if total > 0:
        importances = importances / total

    pairs = sorted(
        zip(feature_names, importances.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    return [
        {"feature": feat, "importance": round(float(imp), 4)}
        for feat, imp in pairs
    ]


def compute_shap_values(
    model: Any,
    X_sample: np.ndarray,
    feature_names: list[str],
    problem_type: str,
) -> list[dict[str, Any]]:
    """Compute SHAP summary values (mean |SHAP|) for each feature."""
    try:
        import shap
        sample = X_sample[:min(200, len(X_sample))]
        explainer = shap.Explainer(model, sample)
        shap_vals = explainer(sample)
        vals = np.abs(shap_vals.values)
        if vals.ndim == 3:
            vals = vals.mean(axis=2)
        mean_shap = vals.mean(axis=0)
        pairs = sorted(
            zip(feature_names, mean_shap.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )[:15]
        return [
            {"feature": feat, "shap_value": round(float(v), 4)}
            for feat, v in pairs
        ]
    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)
        return []


def generate_plain_english_explanation(
    best_model_name: str,
    scores: dict[str, float],
    feature_importances: list[dict[str, Any]],
    problem_type: str,
    dataset_shape: dict[str, int],
    preprocessing_suggestions: list[str],
) -> str:
    """
    Generate a plain-English explanation of the ML results.
    Uses rule-based templates — no external LLM required.
    """
    rows = dataset_shape.get("rows", 0)
    cols = dataset_shape.get("columns", 0)

    intro = (
        f"Your dataset contains {rows:,} rows and {cols} columns. "
        f"The system identified this as a **{problem_type}** problem and automatically "
        f"cleaned, encoded, and scaled the data before training."
    )

    if problem_type == "classification":
        primary_metric = "F1-score"
        primary_value = scores.get("f1_weighted", scores.get("accuracy", 0))
        accuracy = scores.get("accuracy", primary_value)
        perf_summary = (
            f"The best model (**{best_model_name}**) achieved an accuracy of "
            f"{accuracy * 100:.1f}% and a weighted F1-score of {primary_value * 100:.1f}%. "
        )
        if primary_value >= 0.9:
            quality = "This is an excellent result."
        elif primary_value >= 0.75:
            quality = "This is a good result with room for improvement."
        else:
            quality = (
                "The model has moderate performance. Consider collecting more data "
                "or engineering better features."
            )
    else:
        rmse = scores.get("rmse", 0)
        r2 = scores.get("r2", 0)
        perf_summary = (
            f"The best model (**{best_model_name}**) achieved an RMSE of {rmse:.4f} "
            f"and an R² of {r2:.4f}. "
        )
        if r2 >= 0.9:
            quality = "This is an excellent fit."
        elif r2 >= 0.7:
            quality = "This is a reasonable fit."
        else:
            quality = (
                "The model explains some variance but may benefit from more data "
                "or feature engineering."
            )

    features_text = ""
    if feature_importances:
        top3 = [fi["feature"] for fi in feature_importances[:3]]
        features_text = (
            f"\n\nThe three most influential features are: **{', '.join(top3)}**. "
            "These variables had the greatest impact on the model's predictions."
        )

    preprocessing_text = ""
    if preprocessing_suggestions:
        preprocessing_text = (
            "\n\nAutomated preprocessing steps applied: "
            + "; ".join(preprocessing_suggestions[:4])
            + "."
        )

    return (
        intro
        + "\n\n"
        + perf_summary
        + quality
        + features_text
        + preprocessing_text
    )
