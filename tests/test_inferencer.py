import logging
import pandas as pd
import numpy as np
import pytest
import tempfile
import yaml
from unittest.mock import patch, MagicMock

from src.inferencer.inferencer import run_inference

logging.basicConfig(level=logging.INFO)


@pytest.fixture
def dummy_input_csv(tmp_path):
    """
    Creates a temporary CSV file with dummy input data.
    """
    df = pd.DataFrame({
        "feature1": [1, 2],
        "feature2": [3, 4],
        "target": [0, 1]
    })
    file_path = tmp_path / "input.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def dummy_output_csv(tmp_path):
    """
    Returns the path for a temporary output CSV file.
    """
    return tmp_path / "output.csv"


@pytest.fixture
def dummy_config_yaml(tmp_path):
    """
    Creates a dummy YAML config file for inference testing.
    """
    config = {
        "raw_features": ["feature1", "feature2", "target"],
        "target": "target",
        "features": {"engineered": []},
        "artifacts": {
            "preprocessing_pipeline": "fake_pp.pkl",
            "model_path": "fake_model.pkl"
        },
        "logging": {
            "log_file": str(tmp_path / "main.log")
        }
    }
    path = tmp_path / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return path


@patch("src.inferencer.inferencer._load_pickle")
@patch("src.inferencer.inferencer.add_engineered_features",
       side_effect=lambda df: df)
@patch("src.inferencer.inferencer.get_output_feature_names",
       return_value=["feature1", "feature2"])
def test_run_inference(
    mock_get_output_names,
    mock_add_engineered,
    mock_load_pickle,
    dummy_input_csv,
    dummy_config_yaml,
    dummy_output_csv,
):
    """
    Tests the full run_inference flow with mocks.
    """
    mock_pipeline = MagicMock()
    mock_pipeline.transform.return_value = [[1, 2], [3, 4]]

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.9]
    mock_model.predict_proba.return_value = np.array([
        [0.9, 0.1],
        [0.1, 0.9]
    ])

    mock_load_pickle.side_effect = [mock_pipeline, mock_model]

    run_inference(
        str(dummy_input_csv),
        str(dummy_config_yaml),
        str(dummy_output_csv)
    )

    output_df = pd.read_csv(dummy_output_csv)
    assert "prediction" in output_df.columns
    assert "prediction_proba" in output_df.columns
    assert len(output_df) == 2


def test_missing_required_column_triggers_exit(
    dummy_config_yaml, dummy_output_csv
):
    """
    Checks that inference exits when required columns are missing.
    """
    df = pd.DataFrame({
        "feature1": [1, 2],  # Missing "feature2"
        "target": [0, 1]
    })
    with tempfile.NamedTemporaryFile(mode="w", delete=False,
                                     suffix=".csv") as f:
        df.to_csv(f.name, index=False)
        input_path = f.name

    with patch("src.inferencer.inferencer._load_pickle"), \
         patch("src.inferencer.inferencer.add_engineered_features",
               side_effect=lambda df: df):
        with pytest.raises(SystemExit):
            run_inference(input_path, str(dummy_config_yaml),
                          str(dummy_output_csv))


@patch("src.inferencer.inferencer._load_pickle")
@patch("src.inferencer.inferencer.add_engineered_features",
       side_effect=lambda df: df)
@patch("src.inferencer.inferencer.get_output_feature_names",
       return_value=["some_other_feature"])
def test_no_engineered_features_exit(
    mock_get_output_names,
    mock_add_engineered,
    mock_load_pickle,
    dummy_input_csv,
    dummy_output_csv,
    dummy_config_yaml,
):
    """
    Tests exit when expected engineered features are missing post-transform.
    """
    mock_pipeline = MagicMock()
    mock_pipeline.transform.return_value = [[1, 2], [3, 4]]
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.9]

    mock_load_pickle.side_effect = [mock_pipeline, mock_model]

    with open(dummy_config_yaml, "r") as f:
        config = yaml.safe_load(f)

    config["features"]["engineered"] = ["feature1"]

    with open(dummy_config_yaml, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    with pytest.raises(SystemExit):
        run_inference(str(dummy_input_csv),
                      str(dummy_config_yaml),
                      str(dummy_output_csv))


@patch("src.inferencer.inferencer._load_pickle")
@patch("src.inferencer.inferencer.add_engineered_features",
       side_effect=lambda df: df)
@patch("src.inferencer.inferencer.get_output_feature_names",
       return_value=["feature1", "feature2"])
def test_model_without_predict_proba(
    mock_get_output_names,
    mock_add_engineered,
    mock_load_pickle,
    dummy_input_csv,
    dummy_config_yaml,
    dummy_output_csv,
):
    """
    Ensures inference works even if model lacks predict_proba().
    """
    mock_pipeline = MagicMock()
    mock_pipeline.transform.return_value = [[1, 2], [3, 4]]

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.9]
    del mock_model.predict_proba  # Simulate missing method

    mock_load_pickle.side_effect = [mock_pipeline, mock_model]

    run_inference(str(dummy_input_csv),
                  str(dummy_config_yaml),
                  str(dummy_output_csv))

    output_df = pd.read_csv(dummy_output_csv)
    assert "prediction" in output_df.columns
    assert "prediction_proba" not in output_df.columns
