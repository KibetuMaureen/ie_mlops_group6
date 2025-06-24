"""
Model training pipeline with:
 - modular components, preprocessing,
 - optional Bayesian optimization,
 - configurable artifact paths.
"""

import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import xgboost as xgb
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from evaluation.evaluator_sklearn import evaluate_model
from features.features import add_engineered_features
from preprocessing.preprocessing import (
    build_preprocessing_pipeline,
    get_output_feature_names,
)

# Optional import for Bayesian Optimization
try:
    from bayes_opt import BayesianOptimization
except ImportError:
    BayesianOptimization = None

# --- Ensure src is on sys.path for absolute imports ---
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "xgboost": xgb.XGBClassifier,
}


def save_artifact(obj, path: str) -> None:
    """
    Save a Python object to disk using pickle.

    Args:
        obj: The object to serialize.
        path: The destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Artifact saved to {path}")


def split_data(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple:
    """
    Split the input dataframe into train, validation, and test sets.

    Args:
        df: Input dataframe.
        config: Configuration dictionary.

    Returns:
        Tuple of feature and target splits.
    """
    target = config["target"]
    split_cfg = config["data_split"]
    X = df.drop(columns=[target], errors='ignore')
    y = df[target]

    test_size = split_cfg.get("test_size", 0.2)
    valid_size = split_cfg.get("valid_size", 0.2)
    random_state = split_cfg.get("random_state", 42)

    if valid_size > 0:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(test_size + valid_size),
            stratify=y,
            random_state=random_state
        )
        rel_valid = valid_size / (test_size + valid_size)
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp, y_temp,
            test_size=rel_valid,
            stratify=y_temp,
            random_state=random_state
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )
        X_valid, y_valid = None, None

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def apply_feature_engineering(X_train, X_valid, X_test):
    """
    Apply feature engineering transformations.

    Args:
        X_train: Training feature dataframe.
        X_valid: Validation feature dataframe or None.
        X_test: Test feature dataframe.

    Returns:
        Tuple of transformed feature dataframes.
    """
    X_train = add_engineered_features(X_train.copy())
    X_valid = (
        add_engineered_features(X_valid.copy())
        if X_valid is not None else None
        )
    X_test = add_engineered_features(X_test.copy())
    return X_train, X_valid, X_test


def preprocess_data(X_train, X_valid, X_test, config):
    """
    Fit and apply preprocessing pipeline to datasets.

    Args:
        X_train: Training features.
        X_valid: Validation features or None.
        X_test: Test features.
        config: Configuration dictionary.

    Returns:
        Tuple of preprocessed dataframes and the pipeline.
    """
    preprocessor = build_preprocessing_pipeline(config)
    X_train_pp = preprocessor.fit_transform(X_train)
    X_valid_pp = (
        preprocessor.transform(X_valid)
        if X_valid is not None else None
        )
    X_test_pp = preprocessor.transform(X_test)

    out_cols = get_output_feature_names(
        preprocessor, X_train.columns.tolist(),
        config
        )

    X_train_pp = pd.DataFrame(
        X_train_pp,
        columns=out_cols,
        index=X_train.index
        )
    X_valid_pp = (
        pd.DataFrame(X_valid_pp, columns=out_cols, index=X_valid.index)
        if X_valid is not None
        else None
    )
    X_test_pp = pd.DataFrame(X_test_pp, columns=out_cols, index=X_test.index)

    return X_train_pp, X_valid_pp, X_test_pp, preprocessor


def select_input_features(X_pp, config):
    """
    Select features from the preprocessed dataframe based on config.

    Args:
        X_pp: Preprocessed dataframe.
        config: Configuration dictionary.

    Returns:
        List of selected input features.
    """
    engineered_features = config.get("features", {}).get("engineered", [])
    input_features = [f for f in engineered_features if f in X_pp.columns]
    if not input_features:
        logger.warning("No engineered features matched. Using all columns.")
        input_features = X_pp.columns.tolist()
    return input_features


def train_model(X_train, y_train, model_type, params):
    """
    Train a model of the specified type.

    Args:
        X_train: Feature matrix.
        y_train: Target vector.
        model_type: Key from MODEL_REGISTRY.
        params: Parameters to pass to the model.

    Returns:
        Trained model.
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
    Run Bayesian Optimization to tune model hyperparameters.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_valid: Validation features.
        y_valid: Validation target.
        model_type: Model type as string.
        bo_cfg: Configuration for Bayesian optimization.

    Returns:
        Dictionary of optimized parameters.
    """
    from sklearn.metrics import recall_score

    if BayesianOptimization is None:
        raise ImportError("bayesian-optimization package not installed.")

    def eval_func(**params):
        params = {
            k: int(v) if k in ["max_depth", "n_estimators",
                               "min_samples_split", "min_samples_leaf"]
            else v
            for k, v in params.items()
        }
        model_cls = MODEL_REGISTRY[model_type](**params)
        model = model_cls.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        return recall_score(y_valid, y_pred)

    pbounds = {k: tuple(v) for k, v in bo_cfg["search_space"].items()}
    optimizer = BayesianOptimization(
        f=eval_func,
        pbounds=pbounds,
        random_state=bo_cfg.get("random_state", 42)
    )
    optimizer.maximize(
        init_points=bo_cfg.get("init_points", 5),
        n_iter=bo_cfg.get("n_iter", 20)
    )
    best_params = optimizer.max["params"]
    return {
        k: int(v) if isinstance(v, float) else v
        for k, v in best_params.items()
        }


def evaluate_and_log(model, X, y, dataset_name):
    """
    Evaluate a model and log results.

    Args:
        model: Trained model.
        X: Feature matrix.
        y: Ground truth labels.
        dataset_name: Name of the dataset (e.g., 'Test', 'Validation').
    """
    y_pred = model.predict(X)
    results = evaluate_model(y, y_pred)
    logger.info(f"{dataset_name} Metrics: {results}")


def run_model_pipeline(df: pd.DataFrame, config: Dict[str, Any]):
    """
    Complete training pipeline including preprocessing, training,
    optional hyperparameter optimization, and evaluation.

    Returns:
        The trained model and the preprocessing pipeline.
    """
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df, config)

    X_train, X_valid, X_test = (
        apply_feature_engineering(X_train, X_valid, X_test)
        )

    # Save feature-engineered splits
    split_path = config.get("artifacts", {}).get("split_path", "data/splits")
    os.makedirs(split_path, exist_ok=True)
    X_train.to_csv(
        os.path.join(split_path, "X_train_engineered.csv"),
        index=False
        )
    X_test.to_csv(
        os.path.join(split_path, "X_test_engineered.csv"),
        index=False
        )
    if X_valid is not None:
        X_valid.to_csv(
            os.path.join(split_path, "X_valid_engineered.csv"),
            index=False
            )
    logger.info(f"Saved feature-engineered splits to {split_path}")

    X_train_pp, X_valid_pp, X_test_pp, preprocessor = preprocess_data(
        X_train, X_valid, X_test, config
    )

    # Save preprocessed splits
    X_train_pp.to_csv(os.path.join(split_path, "X_train.csv"), index=False)
    X_test_pp.to_csv(os.path.join(split_path, "X_test.csv"), index=False)
    pd.Series(y_train, name="target").to_csv(
        os.path.join(split_path, "y_train.csv"),
        index=False
        )
    pd.Series(y_test, name="target").to_csv(
        os.path.join(split_path, "y_test.csv"),
        index=False
        )
    if X_valid_pp is not None:
        X_valid_pp.to_csv(os.path.join(split_path, "X_valid.csv"), index=False)
        pd.Series(y_valid, name="target").to_csv(
            os.path.join(split_path, "y_valid.csv"),
            index=False
            )
    logger.info(f"Saved preprocessed splits to {split_path}")

    # Save preprocessing pipeline
    save_artifact(
        preprocessor,
        config.get("artifacts", {}).get(
            "preprocessing_pipeline", "models/preprocessing_pipeline.pkl")
    )

    # --- Train Model ---
    model_config = config["model"]
    model_type = model_config["name"]
    model_params = model_config.get("params", {})

    # Optional: Bayesian Optimization
    bo_cfg = model_config.get("bayesian_optimization")
    if bo_cfg and bo_cfg["enabled"] and X_valid_pp is not None:
        logger.info("Starting Bayesian Optimization...")
        best_params = bayesian_optimize(
            X_train_pp.values, y_train,
            X_valid_pp.values, y_valid,
            model_type,
            bo_cfg
        )
        model_params.update(best_params)
        logger.info(f"Optimized parameters: {best_params}")

    # Train with final or optimized params
    model = train_model(X_train_pp.values, y_train, model_type, model_params)

    try:
        save_artifact(
            model,
            config.get("artifacts", {}).get("model_path", "models/model.pkl")
        )

        # --- Evaluate and Log ---
        logger.info("Evaluating model performance...")
        if X_valid_pp is not None:
            evaluate_and_log(model, X_valid_pp.values, y_valid, "Validation")
        evaluate_and_log(model, X_test_pp.values, y_test, "Test")

        return model, preprocessor

    except Exception as e:  # pylint: disable=broad-except
        logger.exception("Model training pipeline failed")
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    try:
        from src.data_loader.data_loader import get_data
        df = get_data(config_path=config_path, data_stage="raw")
    except ImportError:
        df = pd.read_csv(config["data_source"]["raw_path"])

    run_model_pipeline(df, config)
