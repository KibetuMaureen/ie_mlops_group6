# model.py (modular and optimized for testing and maintainability)

import os
import sys
import yaml
import logging
import pickle
from typing import Dict, Any, Tuple

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb

from src.preprocessing.preprocessing import build_preprocessing_pipeline, get_output_feature_names
from src.features.features import add_engineered_features
from src.evaluation.evaluator_sklearn import evaluate_model

# Optional BayesianOptimization import
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

def save_artifact(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Artifact saved to {path}")

def split_data(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple:
    raw_features = config.get("raw_features", [])
    target = config["target"]
    split_cfg = config["data_split"]
    input_features_raw = [f for f in raw_features if f != target]

    X = df[input_features_raw]
    y = df[target]
    test_size = split_cfg.get("test_size", 0.2)
    valid_size = split_cfg.get("valid_size", 0.2)
    random_state = split_cfg.get("random_state", 42)

    if valid_size > 0:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + valid_size), stratify=y, random_state=random_state
        )
        rel_valid = valid_size / (test_size + valid_size)
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp, y_temp, test_size=rel_valid, stratify=y_temp, random_state=random_state
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        X_valid, y_valid = None, None

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def apply_feature_engineering(X_train, X_valid, X_test):
    X_train = add_engineered_features(X_train.copy())
    X_valid = add_engineered_features(X_valid.copy()) if X_valid is not None else None
    X_test = add_engineered_features(X_test.copy())
    return X_train, X_valid, X_test

def preprocess_data(X_train, X_valid, X_test, config):
    preprocessor = build_preprocessing_pipeline(config)
    X_train_pp = preprocessor.fit_transform(X_train)
    X_valid_pp = preprocessor.transform(X_valid) if X_valid is not None else None
    X_test_pp = preprocessor.transform(X_test)

    out_cols = get_output_feature_names(preprocessor, X_train.columns.tolist(), config)
    X_train_pp = pd.DataFrame(X_train_pp, columns=out_cols, index=X_train.index)
    X_valid_pp = pd.DataFrame(X_valid_pp, columns=out_cols, index=X_valid.index) if X_valid is not None else None
    X_test_pp = pd.DataFrame(X_test_pp, columns=out_cols, index=X_test.index)

    return X_train_pp, X_valid_pp, X_test_pp, preprocessor

def select_input_features(X_pp, config):
    engineered_features = config.get("features", {}).get("engineered", [])
    input_features = [f for f in engineered_features if f in X_pp.columns]
    if not input_features:
        logger.warning("No engineered features matched. Using all columns.")
        input_features = X_pp.columns.tolist()
    return input_features

def train_model(X_train, y_train, model_type, params):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}")
    model_cls = MODEL_REGISTRY[model_type]
    model = model_cls(**params)
    model.fit(X_train, y_train)
    logger.info(f"Trained model: {model_type}")
    return model

def bayesian_optimize(X_train, y_train, X_valid, y_valid, model_type, bo_cfg):
    from sklearn.metrics import recall_score
    if BayesianOptimization is None:
        raise ImportError("bayesian-optimization package not installed.")

    def eval_func(**params):
        params = {k: int(v) if k in ["max_depth", "n_estimators", "min_samples_split", "min_samples_leaf"] else v
                  for k, v in params.items()}
        model_cls = MODEL_REGISTRY[model_type](**params)
        model = model_cls.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        return recall_score(y_valid, y_pred)

    pbounds = {k: tuple(v) for k, v in bo_cfg["search_space"].items()}
    optimizer = BayesianOptimization(f=eval_func, pbounds=pbounds,
                                     random_state=bo_cfg.get("random_state", 42))
    optimizer.maximize(init_points=bo_cfg.get("init_points", 5), n_iter=bo_cfg.get("n_iter", 20))
    best_params = optimizer.max["params"]
    return {k: int(v) if isinstance(v, float) else v for k, v in best_params.items()}

def evaluate_and_log(model, X, y, dataset_name):
    y_pred = model.predict(X)
    results = evaluate_model(y, y_pred)
    logger.info(f"{dataset_name} Metrics: {results}")

def run_model_pipeline(df: pd.DataFrame, config: Dict[str, Any]):
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df, config)
    X_train, X_valid, X_test = apply_feature_engineering(X_train, X_valid, X_test)
    X_train_pp, X_valid_pp, X_test_pp, preprocessor = preprocess_data(X_train, X_valid, X_test, config)

    input_features = select_input_features(X_train_pp, config)
    X_train_pp = X_train_pp[input_features]
    X_valid_pp = X_valid_pp[input_features] if X_valid_pp is not None else None
    X_test_pp = X_test_pp[input_features]

    save_artifact(preprocessor, config.get("artifacts", {}).get("preprocessing_pipeline", "models/preprocessing_pipeline.pkl"))

    model_type = config["model"].get("active", "decision_tree")
    model_cfg = config["model"][model_type]
    params = model_cfg.get("params", {})

    if model_cfg.get("bayesian_optimization", {}).get("enabled", False) and X_valid_pp is not None:
        best_params = bayesian_optimize(X_train_pp.values, y_train, X_valid_pp.values, y_valid, model_type,
                                        model_cfg["bayesian_optimization"])
        params.update(best_params)

    model = train_model(X_train_pp.values, y_train, model_type, params)
    save_artifact(model, config.get("artifacts", {}).get("model_path", "models/model.pkl"))

    if X_valid_pp is not None:
        evaluate_and_log(model, X_valid_pp.values, y_valid, "Validation")
    evaluate_and_log(model, X_test_pp.values, y_test, "Test")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    try:
        from src.data_loader.data_loader import get_data
        df = get_data(config_path=config_path, data_stage="raw")
    except ImportError:
        df = pd.read_csv(config["data_source"]["raw_path"])

    run_model_pipeline(df, config)
