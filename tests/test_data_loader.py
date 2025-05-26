import os

import pytest
import pandas as pd
from src.data_loader.data_loader import load_config, load_data, get_data

CONFIG_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '..', 'config.yaml'
    )
)


@pytest.fixture
def test_load_config():
    """Fixture to load configuration for tests."""
    return load_config(CONFIG_PATH)


def test_debug_config_contents():
    """Test to print and check config contents."""
    config = load_config(CONFIG_PATH)
    print("Config loaded:", config)
    assert "data_source" in config


def test_load_config_success():
    """Test successful config loading."""
    config = load_config(CONFIG_PATH)
    assert isinstance(config, dict)
    assert "data_source" in config


def test_load_data_success():
    """Test successful data loading from config."""
    config = load_config(CONFIG_PATH)
    raw_path = config["data_source"]["raw_path"]
    delimiter = config["data_source"].get("delimiter", ",")
    header = config["data_source"].get("header", 0)
    encoding = config["data_source"].get("encoding", "utf-8")

    if not os.path.exists(raw_path):
        pytest.skip(f"Data file not found at path: {raw_path}")

    df = load_data(
        path=raw_path,
        delimiter=delimiter,
        header=header,
        encoding=encoding,
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_get_data_success():
    """Test get_data returns a non-empty DataFrame."""
    df = get_data(config_path=CONFIG_PATH, data_stage="raw")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_load_config_missing():
    """Test loading a missing config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")


def test_load_data_missing_file():
    """Test loading a missing data file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_data(path="nonexistent.csv")
