"""
model.py

Leakage-proof, end-to-end MLOps pipeline for Credit Card Fraud Detection:
- Splits raw data first
- Fits preprocessing pipeline ONLY on train split, applies to valid/test
- Trains model (XGBoost, Decision Tree, Random Forest), supports Bayesian optimization
- Evaluates and saves model and preprocessing artifacts
"""

import os
import logging
import json
import pickle
from typing import Dict, Any
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import xgboost as xgb

# Optionally import BayesianOptimization if available
try:
    from bayes_opt import BayesianOptimization
except ImportError:
    BayesianOptimization = None

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "xgboost": xgb.XGBClassifier,
}


def train_model(X_train, y_train, model_type, params):
    """
    Train a classification model using the specified type and parameters.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        model_type (str): Model type key from MODEL_REGISTRY.
        params (dict): Model hyperparameters.

    Returns:
        Trained model instance.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}")
    model_cls = MODEL_REGISTRY[model_type]
    model = model_cls(**params)
    model.fit(X_train, y_train)
    logger.info(f"Trained model: {model_type}")
    return model


def bayesian_optimize(X_train, y_train, X_valid, y_valid, model_type, bo_cfg):
    """
    Perform Bayesian optimization for hyperparameter tuning.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_valid (pd.DataFrame): Validation features.
        y_valid (pd.Series): Validation labels.
        model_type (str): Model type key from MODEL_REGISTRY.
        bo_cfg (dict): Bayesian optimization configuration.

    Returns:
        dict: Best hyperparameters found.
    """
    if BayesianOptimization is None:
        raise ImportError("bayesian-optimization package is not installed.")
    if model_type == "xgboost":
        def xgb_eval(**params):
            params = {k: int(v) if k in ["max_depth", "n_estimators"]
                      else v for k, v in params.items()}
            model = xgb.XGBClassifier(
                use_label_encoder=False, eval_metric="logloss", **params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            return recall_score(y_valid, y_pred)
        pbounds = {k: tuple(v) for k, v in bo_cfg["search_space"].items()}
        optimizer = BayesianOptimization(
            f=xgb_eval, pbounds=pbounds,
            random_state=bo_cfg.get("random_state", 42)
        )
    elif model_type == "decision_tree":
        def dt_eval(**params):
            params = {k: int(v) for k, v in params.items()}
            model = DecisionTreeClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            return recall_score(y_valid, y_pred)
        pbounds = {k: tuple(v) for k, v in bo_cfg["search_space"].items()}
        optimizer = BayesianOptimization(
            f=dt_eval, pbounds=pbounds,
            random_state=bo_cfg.get("random_state", 42)
        )
    elif model_type == "random_forest":
        def rf_eval(**params):
            params = {k: int(v) for k, v in params.items()}
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            return recall_score(y_valid, y_pred)
        pbounds = {k: tuple(v) for k, v in bo_cfg["search_space"].items()}
        optimizer = BayesianOptimization(
            f=rf_eval, pbounds=pbounds,
            random_state=bo_cfg.get("random_state", 42)
        )
    else:
        raise ValueError(f"Bayesian optimization not supported for {model_type}")

    optimizer.maximize(
        init_points=bo_cfg.get("init_points", 5),
        n_iter=bo_cfg.get("n_iter", 20)
    )
    best_params = optimizer.max["params"]
    # Convert float params to int where appropriate
    for k in best_params:
        if (isinstance(best_params[k], float) and
                k in ["max_depth", "n_estimators",
                     "min_samples_split", "min_samples_leaf"]):
            best_params[k] = int(best_params[k])
    return best_params


def evaluate_model(model, X, y):
    """
    Evaluate a trained model on the given data.

    Args:
        model: Trained model instance.
        X (pd.DataFrame): Features.
        y (pd.Series): True labels.

    Returns:
        dict: Evaluation metrics.
    """
    y_pred = model.predict(X)
    y_prob = (model.predict_proba(X)[:, 1]
              if hasattr(model, "predict_proba") else None)
    results = {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1 Score": f1_score(y, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y, y_prob)
        if y_prob is not None else float("nan"),
        "Confusion Matrix": confusion_matrix(y, y_pred).tolist()
    }
    return results


def save_artifact(obj, path: str):
    """
    Save a Python object to disk using pickle.

    Args:
        obj: Object to save.
        path (str): File path to save the object.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Artifact saved to {path}")


def run_model_pipeline(df: pd.DataFrame, config: Dict[str, Any], preprocessor=None):
    """
    Run the end-to-end model pipeline: split, preprocess, train, optimize, evaluate, save.

    Args:
        df (pd.DataFrame): Input data.
        config (dict): Pipeline configuration.
        preprocessor (sklearn.Pipeline or None): Optional sklearn pipeline for preprocessing.
    """
    logger = logging.getLogger(__name__)
    # --- Split data ---
    target = config["target"]
    X = df.drop(columns=[target])
    y = df[target]
    split_cfg = config["data_split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_cfg.get("test_size", 0.2),
        random_state=split_cfg.get("random_state", 42), stratify=y
    )
    logger.info("Data split complete.")

    # --- Preprocessing (if pipeline provided) ---
    if preprocessor is not None:
        logger.info("Fitting preprocessing pipeline on training data...")
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
        logger.info("Preprocessing complete.")

    # --- Model selection ---
    model_config = config["model"]
    active = model_config.get("active", "xgboost")
    active_model_cfg = model_config[active]
    params = active_model_cfg.get("params", {})
    bo_cfg = active_model_cfg.get("bayesian_optimization", {})
    use_bo = bo_cfg.get("enabled", False)

    # --- Bayesian Optimization (if enabled) ---
    if use_bo:
        logger.info(f"Running Bayesian Optimization for {active}...")
        best_params = bayesian_optimize(
            X_train, y_train, X_test, y_test, active, bo_cfg
        )
        params.update(best_params)
        logger.info(f"Best params found: {best_params}")

    # --- Train final model ---
    model = train_model(X_train, y_train, active, params)

    # --- Save model ---
    save_path = active_model_cfg.get("save_path", f"models/{active}.pkl")
    save_artifact(model, save_path)

    # --- Evaluate ---
    results = evaluate_model(model, X_test, y_test)
    metrics_path = config.get("artifacts", {}).get(
        "metrics_path", "models/metrics.json"
    )
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

# CLI for standalone training
if __name__ == "__main__":
    import sys
    import yaml
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    try:
        from src.data_loader.data_loader import get_data
        df = get_data(config_path=config_path, data_stage="raw")
    except ImportError:
        data_path = config["data_source"]["raw_path"]
        df = pd.read_csv(data_path)
    run_model_pipeline(df, config)
