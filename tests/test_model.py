import pytest
import numpy as np
import pandas as pd
import pickle

from src.model.model import (
    train_model, save_artifact,
    run_model_pipeline,
    MODEL_REGISTRY,
    split_data, apply_feature_engineering,
    preprocess_data, select_input_features,
    evaluate_and_log
)


@pytest.fixture
def toy_df():
    """Fixture that returns a toy pandas DataFrame for testing."""
    return pd.DataFrame({
        "trans_date_trans_time": ["1/1/2019 0:00"] * 10,
        "category": ["grocery_pos", "shopping_net"] * 5,
        "amt": [100 + i for i in range(10)],
        "gender": ["F"] * 5 + ["M"] * 5,
        "city": ["CityA"] * 10,
        "state": ["ST"] * 10,
        "zip": [12345] * 10,
        "lat": [40.0] * 10,
        "long": [-75.0] * 10,
        "city_pop": [5000] * 10,
        "job": ["Engineer"] * 10,
        "dob": ["1/1/1980"] * 10,
        "merch_lat": [39.9] * 10,
        "merch_long": [-75.1] * 10,
        "is_fraud": [0, 1] * 5
    })


@pytest.fixture
def minimal_config(tmp_path):
    """Fixture that returns a minimal model config dictionary for testing."""
    return {
        "raw_features": [
            "trans_date_trans_time", "category", "amt",
            "gender", "city", "state", "zip",
            "lat", "long", "city_pop", "job", "dob",
            "merch_lat", "merch_long"
        ],
        "target": "is_fraud",
        "data_split": {
            "test_size": 0.2,
            "valid_size": 0.2,
            "random_state": 42
        },
        "features": {
            "engineered": ["amt", "city_pop"]
        },
        "preprocessing": {
            "label_encode": ["category", "gender"],
            "numeric": ["amt", "city_pop"]
        },
        "model": {
            "active": "decision_tree",
            "decision_tree": {
                "params": {"max_depth": 3},
                "bayesian_optimization": {
                    "enabled": True,
                    "search_space": {
                        "max_depth": (2, 10)
                    }
                }
            }
        },
        "artifacts": {
            "preprocessing_pipeline": str(tmp_path / "pp.pkl"),
            "model_path": str(tmp_path / "model.pkl")
        },
        "data_source": {
            "raw_path": str(tmp_path / "dummy.csv")
        }
    }


def test_train_model():
    """Test that train_model trains and returns a model with predict method."""
    X = np.random.rand(10, 3)
    y = [0, 1] * 5
    model = train_model(
        X, y,
        model_type="decision_tree",
        params={"max_depth": 2}
        )
    assert hasattr(model, "predict")


def test_save_artifact(tmp_path):
    """Test that save_artifact writes an object to disk correctly."""
    obj = {"key": "value"}
    path = tmp_path / "artifact.pkl"
    save_artifact(obj, str(path))
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    assert loaded == obj


def test_run_model_pipeline(monkeypatch, toy_df, minimal_config):
    """Test that run_model_pipeline runs without errors (saving is mocked)."""
    monkeypatch.setattr(
        "src.model.model.save_artifact",
        lambda *args,
        **kwargs: None
        )
    run_model_pipeline(toy_df, minimal_config)


def test_split_data_shapes(toy_df, minimal_config):
    """Test that split_data returns correct shapes for all splits."""
    X_train, X_valid, X_test, y_train, y_valid, y_test = (
        split_data(
            toy_df,
            minimal_config
            ))
    assert len(X_train) + len(X_valid) + len(X_test) == len(toy_df)
    assert len(y_train) + len(y_valid) + len(y_test) == len(toy_df)


def test_apply_feature_engineering_returns(toy_df):
    """
    Test that apply_feature_engineering
        - returns three DataFrames for data splits.
    """
    df = toy_df.drop(columns=["is_fraud"])
    train, valid, test = df[:6], df[6:8], df[8:]
    train_fe, valid_fe, test_fe = apply_feature_engineering(train, valid, test)
    assert isinstance(train_fe, pd.DataFrame)
    assert isinstance(valid_fe, pd.DataFrame)
    assert isinstance(test_fe, pd.DataFrame)


def test_preprocess_data_returns(toy_df, minimal_config):
    """
    Test that preprocess_data

    Returns processed DataFrames and a preprocessor object.
    """
    df = toy_df.drop(columns=["is_fraud"])
    train, valid, test = df[:6], df[6:8], df[8:]
    train_fe, valid_fe, test_fe = apply_feature_engineering(train, valid, test)
    train_pp, valid_pp, test_pp, preprocessor = preprocess_data(
        train_fe, valid_fe, test_fe, minimal_config
    )
    assert isinstance(train_pp, pd.DataFrame)
    assert isinstance(valid_pp, pd.DataFrame)
    assert isinstance(test_pp, pd.DataFrame)
    assert preprocessor is not None


def test_select_input_features_defaults(toy_df, minimal_config):

    """Test that select_input_features returns list of string feature names."""

    df = toy_df.drop(columns=["is_fraud"])
    df = apply_feature_engineering(df, df, df)[0]
    train, valid, test = df[:6], df[6:8], df[8:]
    train_pp, valid_pp, test_pp, _ = preprocess_data(train, valid, test, minimal_config)
    features = select_input_features(train_pp, minimal_config)
    assert isinstance(features, list)
    assert all(isinstance(f, str) for f in features)


def test_evaluate_and_log(toy_df):
    """Test that evaluate_and_log runs without error on a fitted model."""
    from sklearn.tree import DecisionTreeClassifier
    X = np.random.rand(10, 3)
    y = [0, 1] * 5
    model = DecisionTreeClassifier().fit(X, y)
    evaluate_and_log(model, X, y, "Test")


def test_model_registry():
    """Test that all models in MODEL_REGISTRY are callable."""
    for model_name in MODEL_REGISTRY:
        assert callable(MODEL_REGISTRY[model_name])
