import os
import yaml
import pytest
import pandas as pd

from src.data_loader.data_loader import (
    load_config,
    load_data,
    get_data,
    load_env,
)

CONFIG_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "config.yaml"
    )
)


def test_load_config_success():
    """Test successful loading of a valid config file."""
    assert os.path.isfile(CONFIG_PATH), f"{CONFIG_PATH} not found"
    config = load_config(CONFIG_PATH)
    assert isinstance(config, dict)
    assert "data_source" in config
    assert config is not None


def test_load_config_file_not_found():
    """Test FileNotFoundError is raised when config file is missing."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")


def test_debug_config_contents():
    """Test expected keys exist in the loaded config."""
    config = load_config(CONFIG_PATH)
    assert config is not None
    assert "data_source" in config
    print("Config loaded:", config)


def test_load_data_success():
    """Test that data loads correctly from a valid config path."""
    config = load_config(CONFIG_PATH)
    data_cfg = config.get("data_source", {})
    raw_path = data_cfg.get("raw_path")
    delimiter = data_cfg.get("delimiter", ",")
    header = data_cfg.get("header", 0)
    encoding = data_cfg.get("encoding", "utf-8")

    if not raw_path or not os.path.exists(raw_path):
        pytest.skip(f"Raw data file not found at: {raw_path}")

    df = load_data(
        path=raw_path,
        delimiter=delimiter,
        header=header,
        encoding=encoding,
    )

    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_load_data_missing_file():
    """Test FileNotFoundError is raised for a nonexistent data file."""
    with pytest.raises(FileNotFoundError):
        load_data(path="nonexistent.csv")


def test_get_data_success():
    """Test successful full data retrieval through get_data()."""
    if not os.path.isfile(CONFIG_PATH):
        pytest.skip(f"Config file missing: {CONFIG_PATH}")

    df = get_data(config_path=CONFIG_PATH, data_stage="raw")
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape[0] > 5
    assert df.select_dtypes(include="number").shape[1] >= 1


def test_get_data_invalid_stage():
    """Test ValueError is raised for an invalid data_stage argument."""
    with pytest.raises(ValueError):
        get_data(config_path=CONFIG_PATH, data_stage="invalid_stage")


def test_load_config_invalid_format(tmp_path):
    """Test ValueError is raised if config file is not a dictionary."""
    bad_config_path = tmp_path / "bad_config.yaml"
    bad_config_path.write_text("[]")  # Invalid format

    with pytest.raises(ValueError):
        load_config(str(bad_config_path))


def test_load_env_executes():
    """Test load_env executes without error even if .env doesn't exist."""
    load_env()  # Should not raise any error


def test_load_data_invalid_path_type():
    """Test ValueError is raised when path is None."""
    with pytest.raises(ValueError):
        load_data(path=None)


def test_get_data_missing_path(tmp_path):
    """Test ValueError is raised when required path keys are missing."""
    bad_config = {
        "data_source": {
            "delimiter": ","
        }
    }
    config_path = tmp_path / "bad_config.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    with pytest.raises(ValueError):
        get_data(config_path=str(config_path), data_stage="raw")


def test_get_data_missing_path_key(tmp_path):
    """Test ValueError when 'raw_path' or 'processed_path' is missing."""
    broken_config = {
        "data_source": {
            "delimiter": ",",
            "header": 0,
            "encoding": "utf-8"
        }
    }
    config_path = tmp_path / "missing_key.yaml"
    config_path.write_text(yaml.safe_dump(broken_config))

    with pytest.raises(ValueError):
        get_data(config_path=str(config_path), data_stage="raw")
