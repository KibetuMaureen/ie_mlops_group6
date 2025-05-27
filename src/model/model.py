import os
import logging
import pickle
from typing import Dict, Any
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb

from src.preprocessing.preprocessing import (
    build_preprocessing_pipeline,
    get_output_feature_names,
)

from src.features.features import add_engineered_features

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
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}")
    model_cls = MODEL_REGISTRY[model_type]
    model = model_cls(**params)
    model.fit(X_train, y_train)
    logger.info(f"Trained model: {model_type}")
    return model


def save_artifact(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Artifact saved to {path}")


def bayesian_optimize(X_train, y_train, X_valid, y_valid, model_type, bo_cfg):
    """
    Perform Bayesian optimization for hyperparameter tuning.
    """
    from sklearn.metrics import recall_score

    if BayesianOptimization is None:
        raise ImportError("bayesian-optimization package is not installed.")
    if model_type == "xgboost":
        def xgb_eval(**params):
            params = {
                k: int(v) if k in ["max_depth", "n_estimators"] else v
                for k, v in params.items()
            }
            model = xgb.XGBClassifier(
                use_label_encoder=False, eval_metric="logloss", **params
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            return recall_score(y_valid, y_pred)

        pbounds = {k: tuple(v) for k, v in bo_cfg["search_space"].items()}
        optimizer = BayesianOptimization(
            f=xgb_eval, pbounds=pbounds, random_state=bo_cfg.get("random_state", 42)
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
            f=dt_eval, pbounds=pbounds, random_state=bo_cfg.get("random_state", 42)
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
            f=rf_eval, pbounds=pbounds, random_state=bo_cfg.get("random_state", 42)
        )
    else:
        raise ValueError(f"Bayesian optimization not supported for {model_type}")

    optimizer.maximize(
        init_points=bo_cfg.get("init_points", 5), n_iter=bo_cfg.get("n_iter", 20)
    )
    best_params = optimizer.max["params"]
    # Convert float params to int where appropriate
    for k in best_params:
        if (
            isinstance(best_params[k], float)
            and k
            in ["max_depth", "n_estimators", "min_samples_split", "min_samples_leaf"]
        ):
            best_params[k] = int(best_params[k])
    return best_params


def run_model_pipeline(df: pd.DataFrame, config: Dict[str, Any]):
    """
    End-to-end, leakage-proof pipeline:
    1. Split raw data into train/valid/test.
    2. Feature engineering on each split.
    3. Fit preprocessing pipeline on train, transform all splits.
    4. Select only engineered/model features.
    5. Train, evaluate, and save model and artifacts.
    """
    # 1. Split data using only raw features
    raw_features = config.get("raw_features", [])
    target = config["target"]
    split_cfg = config["data_split"]
    input_features_raw = [f for f in raw_features if f != target]

    X = df[input_features_raw]
    y = df[target]
    test_size = split_cfg.get("test_size", 0.2)
    valid_size = split_cfg.get("valid_size", 0.2)
    random_state = split_cfg.get("random_state", 42)

    # Robust splitting logic
    if valid_size > 0:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=(test_size + valid_size),
            random_state=random_state,
            stratify=y,
        )
        rel_valid = valid_size / (test_size + valid_size)
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp, y_temp, test_size=rel_valid, random_state=random_state, stratify=y_temp
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_valid, y_valid = None, None

    # 2. Feature engineering on each split
    X_train = add_engineered_features(X_train.copy())
    if X_valid is not None:
        X_valid = add_engineered_features(X_valid.copy())
    X_test = add_engineered_features(X_test.copy())

    # Save raw data splits
    splits_dir = config.get("artifacts", {}).get("splits_dir", "data/splits")
    os.makedirs(splits_dir, exist_ok=True)
    X_train.assign(**{target: y_train}).to_csv(
        os.path.join(splits_dir, "train.csv"), index=False
    )
    if X_valid is not None:
        X_valid.assign(**{target: y_valid}).to_csv(
            os.path.join(splits_dir, "valid.csv"), index=False
        )
    X_test.assign(**{target: y_test}).to_csv(
        os.path.join(splits_dir, "test.csv"), index=False
    )

    # 3. Fit preprocessing pipeline on train, transform all splits
    preprocessor = build_preprocessing_pipeline(config)
    X_train_pp = preprocessor.fit_transform(X_train)
    if X_valid is not None:
        X_valid_pp = preprocessor.transform(X_valid)
    X_test_pp = preprocessor.transform(X_test)

    # 4. Rebuild DataFrames with correct column names and index
    out_cols = get_output_feature_names(
        preprocessor, X_train.columns.tolist(), config
    )
    X_train_pp = pd.DataFrame(X_train_pp, columns=out_cols, index=X_train.index)
    if X_valid is not None:
        X_valid_pp = pd.DataFrame(X_valid_pp, columns=out_cols, index=X_valid.index)
    X_test_pp = pd.DataFrame(X_test_pp, columns=out_cols, index=X_test.index)

    # 5. Use only selected engineered features
    engineered_features = config.get("features", {}).get("engineered", [])
    input_features = [f for f in engineered_features if f in X_train_pp.columns]
    if not input_features:
        logger.warning(
            "No engineered features matched. Using all columns as input features."
        )
        input_features = X_train_pp.columns.tolist()
    X_train_pp = X_train_pp[input_features]
    if X_valid is not None:
        X_valid_pp = X_valid_pp[input_features]
    X_test_pp = X_test_pp[input_features]

    # Save processed data splits (ensure is_fraud is aligned with index!)
    processed_dir = config.get("artifacts", {}).get("processed_dir", "data/processed")
    os.makedirs(processed_dir, exist_ok=True)
    X_train_pp[target] = y_train
    X_train_pp.to_csv(
        os.path.join(processed_dir, "train_processed.csv"), index=False
    )
    if X_valid is not None:
        X_valid_pp[target] = y_valid
        X_valid_pp.to_csv(
            os.path.join(processed_dir, "valid_processed.csv"), index=False
        )
    X_test_pp[target] = y_test
    X_test_pp.to_csv(os.path.join(processed_dir, "test_processed.csv"), index=False)

    # Save preprocessing pipeline
    preproc_path = config.get("artifacts", {}).get(
        "preprocessing_pipeline", "models/preprocessing_pipeline.pkl"
    )
    save_artifact(preprocessor, preproc_path)

    # Train model
    model_config = config["model"]
    active = model_config.get("active", "decision_tree")
    active_model_cfg = model_config[active]
    model_type = active
    params = active_model_cfg.get("params", {})
    bo_cfg = active_model_cfg.get("bayesian_optimization", {})
    use_bo = bo_cfg.get("enabled", False)

    if use_bo and X_valid is not None:
        logger.info(f"Running Bayesian Optimization for {active}...")
        best_params = bayesian_optimize(
            X_train_pp.values,
            y_train,
            X_valid_pp.values,
            y_valid,
            model_type,
            bo_cfg,
        )
        params.update(best_params)
        logger.info(f"Best params found: {best_params}")

    model = train_model(X_train_pp.values, y_train, model_type, params)

    # Save model artifact
    model_path = config.get("artifacts", {}).get("model_path", "models/model.pkl")
    save_artifact(model, model_path)
    algo_model_path = model_config.get(active, {}).get(
        "save_path", f"models/{active}.pkl"
    )
    save_artifact(model, algo_model_path)


# CLI for standalone training
if __name__ == "__main__":
    import sys
    import yaml
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
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
