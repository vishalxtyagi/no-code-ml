"""Train multiple models with optional Optuna hyperparameter tuning."""
import logging
import warnings
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

def _classification_models() -> dict[str, Any]:
    try:
        import lightgbm as lgb
        import xgboost as xgb
        return {
            "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0),
            "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        }
    except ImportError:
        return {
            "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        }


def _regression_models() -> dict[str, Any]:
    try:
        import lightgbm as lgb
        import xgboost as xgb
        return {
            "Ridge": Ridge(random_state=42),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
            "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        }
    except ImportError:
        return {
            "Ridge": Ridge(random_state=42),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        }


# ---------------------------------------------------------------------------
# Optuna tuning
# ---------------------------------------------------------------------------

def _tune_best_model(
    model_name: str,
    problem_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 20,
) -> Any:
    """Use Optuna to tune the best model and return re-fitted model."""

    def objective(trial: optuna.Trial) -> float:
        if problem_type == "classification":
            if model_name == "RandomForest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                }
                model = RandomForestClassifier(**params, random_state=42)
            elif model_name == "GradientBoosting":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 2, 8),
                }
                model = GradientBoostingClassifier(**params, random_state=42)
            elif model_name == "XGBoost":
                import xgboost as xgb
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 2, 8),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                }
                model = xgb.XGBClassifier(**params, random_state=42, eval_metric="logloss", verbosity=0)
            elif model_name == "LightGBM":
                import lightgbm as lgb
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                }
                model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
            else:
                model = LogisticRegression(
                    C=trial.suggest_float("C", 1e-3, 10, log=True), max_iter=500, random_state=42
                )
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring="f1_weighted")
        else:
            if model_name == "RandomForest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                }
                model = RandomForestRegressor(**params, random_state=42)
            elif model_name == "GradientBoosting":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 2, 8),
                }
                model = GradientBoostingRegressor(**params, random_state=42)
            elif model_name == "XGBoost":
                import xgboost as xgb
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 2, 8),
                }
                model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
            elif model_name == "LightGBM":
                import lightgbm as lgb
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                }
                model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
            else:
                model = Ridge(
                    alpha=trial.suggest_float("alpha", 1e-3, 100, log=True), random_state=42
                )
            scores = -cross_val_score(model, X_train, y_train, cv=3, scoring="neg_root_mean_squared_error")
        return float(scores.mean())

    direction = "maximize" if problem_type == "classification" else "minimize"
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    logger.info("Best params for %s: %s", model_name, best_params)

    # Re-create model with best params
    if problem_type == "classification":
        if model_name == "RandomForest":
            return RandomForestClassifier(**best_params, random_state=42)
        if model_name == "GradientBoosting":
            return GradientBoostingClassifier(**best_params, random_state=42)
        if model_name == "XGBoost":
            import xgboost as xgb
            return xgb.XGBClassifier(**best_params, random_state=42, eval_metric="logloss", verbosity=0)
        if model_name == "LightGBM":
            import lightgbm as lgb
            return lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)
        return LogisticRegression(**best_params, max_iter=500, random_state=42)
    else:
        if model_name == "RandomForest":
            return RandomForestRegressor(**best_params, random_state=42)
        if model_name == "GradientBoosting":
            return GradientBoostingRegressor(**best_params, random_state=42)
        if model_name == "XGBoost":
            import xgboost as xgb
            return xgb.XGBRegressor(**best_params, random_state=42, verbosity=0)
        if model_name == "LightGBM":
            import lightgbm as lgb
            return lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1)
        return Ridge(**best_params, random_state=42)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    problem_type: str,
    tune_best: bool = True,
) -> tuple[dict[str, Any], dict[str, dict[str, float]], str]:
    """
    Train all candidate models, evaluate on test set.
    Returns (trained_models, scores, best_model_name).
    """
    from backend.evaluator.evaluator import score_model

    models = (
        _classification_models()
        if problem_type == "classification"
        else _regression_models()
    )

    trained: dict[str, Any] = {}
    scores: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        logger.info("Training %s ...", name)
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            model_scores = score_model(y_test, preds, problem_type)
            trained[name] = model
            scores[name] = model_scores
            logger.info("%s scores: %s", name, model_scores)
        except Exception as exc:
            logger.warning("Failed to train %s: %s", name, exc)

    if not scores:
        raise RuntimeError("All models failed to train.")

    # Select best model
    if problem_type == "classification":
        best_name = max(scores, key=lambda n: scores[n].get("f1_weighted", 0.0))
    else:
        best_name = min(scores, key=lambda n: scores[n].get("rmse", float("inf")))

    logger.info("Best model before tuning: %s", best_name)

    # Optionally tune the best model
    if tune_best and best_name in trained:
        logger.info("Tuning %s with Optuna ...", best_name)
        tuned_model = _tune_best_model(best_name, problem_type, X_train, y_train, n_trials=20)
        tuned_model.fit(X_train, y_train)
        preds = tuned_model.predict(X_test)
        tuned_scores = score_model(y_test, preds, problem_type)
        trained[best_name] = tuned_model
        scores[best_name] = tuned_scores
        logger.info("Tuned %s scores: %s", best_name, tuned_scores)

    return trained, scores, best_name
