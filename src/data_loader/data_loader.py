"""
data_loader.py

Modular data ingestion utility for CSV and Excel files.
- Loads configuration from config.yaml
"""

import os
import logging

import pandas as pd
import yaml
from dotenv import load_dotenv
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config2.yaml") -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(
            f"Config file must be a YAML mapping (dict), "
            f"got {type(config).__name__}"
        )
    return config


def load_env(env_path: str = ".env"):
    """
    Load environment variables from a .env file.

    Args:
        env_path (str): Path to the .env file.
    """
    load_dotenv(dotenv_path=env_path, override=True)


def load_data(
    path: str,
    delimiter: str = ",",
    header: int = 0,
    encoding: str = "utf-8"
) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        path (str): Path to the data file.
        delimiter (str): Delimiter used in the file.
        header (int): Row number to use as column names.
        encoding (str): File encoding.

    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        ValueError: If path is invalid.
        FileNotFoundError: If file does not exist.
        Exception: If loading fails.
    """
    if not path or not isinstance(path, str):
        logger.error("No valid data path specified in configuration.")
        raise ValueError("No valid data path specified in configuration.")
    if not os.path.isfile(path):
        logger.error("Data file does not exist: %s", path)
        raise FileNotFoundError(f"Data file not found: {path}")
    try:
        df = pd.read_csv(
            path,
            delimiter=delimiter,
            header=header,
            encoding=encoding
        )
        logger.info("Loaded data from %s, shape=%s", path, df.shape)
        return df
    except Exception as e:
        logger.exception("Failed to load data: %s", e)
        raise


def get_data(
    config_path: str = "config.yaml",
    env_path: str = ".env",
    data_stage: str = "raw"
) -> pd.DataFrame:
    """
    Get data based on configuration and data stage.

    Args:
        config_path (str): Path to config YAML.
        env_path (str): Path to .env file.
        data_stage (str): 'raw' or 'processed'.

    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        ValueError: If data_stage is unknown or path is invalid.
    """
    load_env(env_path)
    config = load_config(config_path)
    data_cfg = config.get("data_source", {})
    if data_stage == "raw":
        path = data_cfg.get("raw_path")
    elif data_stage == "processed":
        path = data_cfg.get("processed_path")
    else:
        logger.error("Unknown data_stage: %s", data_stage)
        raise ValueError(f"Unknown data_stage: {data_stage}")
    if not path or not isinstance(path, str):
        logger.error(
            "No valid data path specified in configuration for "
            "data_stage='%s'.",
            data_stage
        )
        raise ValueError(
            f"No valid data path specified in configuration for "
            f"data_stage='{data_stage}'."
        )
    base_dir = Path(config_path).resolve().parent
    resolved_path = (
        base_dir / path).resolve() if not Path(path).is_absolute() else Path(path)
    
    df = load_data(
        path=str(resolved_path),
        delimiter=data_cfg.get("delimiter", ","),
        header=data_cfg.get("header", 0),
        encoding=data_cfg.get("encoding", "utf-8"),
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    try:
        df = get_data(data_stage="raw")
        logging.info("Data loaded successfully. Shape: %s", df.shape)
    except Exception as e:
        logging.exception("Failed to load data: %s", e)
