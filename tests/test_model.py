import pytest
import numpy as np
import pandas as pd
import pickle

from src.model.model import (
    train_model, save_artifact,
    run_model_pipeline,
    MODEL_REGISTRY, bayesian_optimize
)

# --- Fixtures for sample data and config ---

@pytest.fixture
def toy_df():
    """Toy DataFrame with only raw features, pipeline will engineer features."""
    df = pd.DataFrame({
        "trans_date_trans_time": [
            "1/1/2019 0:00", "1/1/2019 0:00", "1/1/2019 0:00",
            "1/1/2019 0:01", "1/1/2019 0:03", "1/1/2019 0:04",
            "1/2/2019 1:06", "1/2/2019 1:47", "1/2/2019 3:05", 
            "1/2/2019 3:38", "1/2/2019 3:55"
        ],
        "category": [
            "misc_net", "grocery_pos", "entertainment",
            "gas_transport", "misc_pos", "gas_transport",
            "grocery_pos", "gas_transport", "grocery_pos", 
            "gas_transport", "grocery_pos"
        ],
        "amt": [
            4.97, 107.23, 220.11, 45, 41.96, 94.63,
            281.06, 11.52, 276.31, 7.03, 275.73
        ],
        "gender": ["F", "F", "M", "M", "M", "F", "M", "F", "F", "M", "F"],
        "city": [
            "Moravian Falls", "Orient", "Malad City", "Boulder", "Doe Hill", "Dublin",
            "Collettsville", "San Antonio", "San Antonio", "Collettsville", "San Antonio"
        ],
        "state": ["NC", "WA", "ID", "MT", "VA", "PA", "NC", "TX", "TX", "NC", "TX"],
        "zip": [28654, 99160, 83252, 59632, 24433, 18917, 28611, 78208, 78208, 28611, 78208],
        "lat": [
            36.0788, 48.8878, 42.1808, 46.2306, 38.4207, 40.375,
            35.9946, 29.44, 29.44, 35.9946, 29.44
        ],
        "long": [
            -81.1781, -118.2105, -112.262, -112.1138, -79.4629, -75.2045,
            -81.7266, -98.459, -98.459, -81.7266, -98.459
        ],
        "city_pop": [3495, 149, 4154, 1939, 99, 2158, 885, 1595797, 1595797, 885, 1595797],
        "job": [
            "Psychologist, counselling", "Special educational needs teacher", "Nature conservation officer",
            "Patent attorney", "Dance movement psychotherapist", "Transport planner",
            "Soil scientist", "Horticultural consultant", "Horticultural consultant", "Soil scientist", "Horticultural consultant"
        ],
        "dob": [
            "3/9/1988", "6/21/1978", "1/19/1962", "1/12/1967", "3/28/1986", "6/19/1961",
            "9/15/1988", "10/28/1960", "10/28/1960", "9/15/1988", "10/28/1960"
        ],
        "merch_lat": [
            36.011293, 49.159047, 43.150704, 47.034331, 38.674999, 40.653382,
            36.430124, 29.819364, 29.273085, 35.909292, 29.786426
        ],
        "merch_long": [
            -82.048315, -118.186462, -112.154481, -112.561071, -78.632459, -76.152667,
            -81.179483, -99.142791, -98.83636, -82.09101, -98.68341
        ],
        "is_fraud": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    })
    return df

@pytest.fixture
def minimal_config(tmp_path):
    return {
        "raw_features": [
            "trans_date_trans_time", "category", "amt", "gender", "city", "state", "zip",
            "lat", "long", "city_pop", "job", "dob", "merch_lat", "merch_long"
        ],
        "target": "is_fraud",
        "data_split": {
            "test_size": 0.33,
            "valid_size": 0.33,
            "random_state": 0
        },
        "features": {
            "engineered": [
                "amt", "city_pop", "hour", "day", "month", "geo_distance",
                "day_of_week", "age", "category", "gender"
            ]
        },
        "preprocessing": {},
        "model": {
            "active": "decision_tree",
            "decision_tree": {"params": {"max_depth": 2}}
        },
        "artifacts": {
            "splits_dir": str(tmp_path / "splits"),
            "processed_dir": str(tmp_path / "processed"),
            "preprocessing_pipeline": str(tmp_path / "preproc.pkl"),
            "model_path": str(tmp_path / "model.pkl"),
            "metrics_path": str(tmp_path / "metrics.json")
        },
        "data_source": {"raw_path": str(tmp_path / "sample_data.csv")}
    }

@pytest.fixture
def Xy():
    X = np.array([
        [-0.118625831, -1.438912977, -0.746609379, 0.543626122, -1.525741615, 0.422626475, -1.006848305],
        [-0.360128486, 0.908057743, 1.518615242, -0.041775026, -0.048472783, 0.877514413, 0.141750923],
        [5.672535246, 0.174629393, 0.272741701, 0.543626122, 0.440885128, 1.33240235, -0.662268536],
        [-0.169753337, 1.201429083, -0.293564455, -0.3344756, -0.574901227, -1.396925275, 0.428900731],
        [-0.237244193, 0.174629393, 0.159480469, 0.543626122, 1.286029337, 0.877514413, -1.408858035],
        [0.825482095, 1.348114753, -0.293564455, -0.041775026, -1.252863405, 0.877514413, 0.83091046],
        [-0.413420793, -0.118741947, -0.633348148, -0.627176174, -0.179097684, 0.422626475, -1.351428073],
        [-0.078067647, 0.761372073, -0.520086917, 0.250925548, 1.657633532, -0.032261463, -0.489978652],
        [-0.296967232, 1.494800423, -0.520086917, 1.421727844, 0.976742071, -1.396925275, 0.199180885],
        [0.218701104, -1.145541637, 1.065570318, -1.212577322, 0.966470485, -1.396925275, -0.375118729],
        [-0.439143723, 1.348114753, -0.85987061, -0.919876748, -1.055917693, 1.33240235, -0.604838575]
    ])
    y = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0])
    return X, y

# --- Tests for model training and registry ---

def test_train_model_supported_types(Xy):
    X, y = Xy
    for mtype in MODEL_REGISTRY:
        model = train_model(X, y, mtype, {})
        assert hasattr(model, "fit") and hasattr(model, "predict")

def test_train_model_unsupported_type(Xy):
    X, y = Xy
    with pytest.raises(ValueError):
        train_model(X, y, "foobar_model", {})

def test_train_model_with_params(Xy):
    X, y = Xy
    model = train_model(X, y, "decision_tree", {"max_depth": 1})
    assert hasattr(model, "max_depth") and model.max_depth == 1

def test_train_model_allows_missing_data_decision_tree():
    X = np.array([[4.97, np.nan, 36.0788, -81.1781], [107.23, 149, 48.8878, -118.2105], [220.11, 4154, 42.1808, -112.262]])
    y = np.array([0, 0, 0])
    model = train_model(X, y, "decision_tree", {})
    preds = model.predict(X)
    assert len(preds) == len(y)

# --- Test for artifact save/load ---

def test_save_and_load_artifact(tmp_path):
    arr = np.array([1, 2, 3])
    path = tmp_path / "artifact.pkl"
    save_artifact(arr, str(path))
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    np.testing.assert_array_equal(arr, loaded)

# --- Bayesian Optimization Test ---

def test_bayesian_optimize_decision_tree(monkeypatch):
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    X_train, X_valid = X[:80], X[80:]
    y_train, y_valid = y[:80], y[80:]
    bo_cfg = {
        "enabled": True,
        "random_state": 0,
        "init_points": 1,
        "n_iter": 1,
        "search_space": {"max_depth": [1, 2]}
    }

    # Skip test if BayesianOptimization not installed
    if bayesian_optimize.__globals__.get("BayesianOptimization") is None:
        pytest.skip("bayesian-optimization not installed")

    best_params = bayesian_optimize(X_train, y_train, X_valid, y_valid, "decision_tree", bo_cfg)
    assert "max_depth" in best_params

# --- Pipeline run test ---
"""
def test_run_model_pipeline_runs_minimal(toy_df, minimal_config, tmp_path):
    # Save toy_df CSV in temp path, and update config to point there
    raw_path = tmp_path / "raw_data.csv"
    toy_df.to_csv(raw_path, index=False)
    minimal_config["data_source"]["raw_path"] = str(raw_path)

    # Load the DataFrame here with date parsing
    df = pd.read_csv(str(raw_path), parse_dates=["trans_date_trans_time"])

    # Run pipeline - pass df and config explicitly
    results = run_model_pipeline(df, minimal_config)

    # Assert model and metrics
    assert "model" in results and results["model"] is not None
    assert "metrics" in results and isinstance(results["metrics"], dict)

    # Check saved artifacts
    assert (tmp_path / "model.pkl").exists()
    assert (tmp_path / "metrics.json").exists()

"""