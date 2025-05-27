"""
main.py

Minimal entry point to check data loading, validation, and run the full ML pipeline with staging support.
"""

import argparse
import sys
import logging
import os
import yaml
from src.data_loader.data_loader import get_data
from src.data_validation.data_validation import validate_schema
from src.model.model import run_model_pipeline

logger = logging.getLogger(__name__)


def setup_logging(logging_config: dict):
    log_file = logging_config.get("log_file", "logs/main.log")
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_format = logging_config.get(
        "format", "%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    date_format = logging_config.get("datefmt", "%Y-%m-%d %H:%M:%S")
    logging.basicConfig(
        filename=log_file,
        level=getattr(logging, logging_config.get("level", "INFO")),
        format=log_format,
        datefmt=date_format,
        filemode="a"
    )
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, logging_config.get("level", "INFO")))
    formatter = logging.Formatter(log_format, date_format)
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Minimal data loading, validation, and pipeline runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file"
    )
    parser.add_argument(
        "--env",
        type=str,
        default=".env",
        help="Path to an optional .env file for environment variables"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["load", "validate", "pipeline", "all"],
        help="Pipeline stage to execute: load, validate, pipeline or all (default: all)"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.exception(f"Failed to load config: {e}")
        sys.exit(1)

    try:
        setup_logging(config.get("logging", {}))
    except Exception as e:
        logger.exception(f"Failed to set up logging: {e}")
        sys.exit(1)

    logger.info("Pipeline started")

    df_raw = None

    # Data loading
    if args.stage in ["load", "all", "validate", "pipeline"]:
        try:
            df_raw = get_data(
                config_path=args.config,
                env_path=args.env,
                data_stage="raw"
            )
            if df_raw is None or not hasattr(df_raw, "shape"):
                logger.error(
                    "Load data failed: get_data did not return valid DataFrame"
                )
                sys.exit(1)
            logger.info(
                "Raw data loaded successfully. Shape: %s",
                df_raw.shape
            )
            print(df_raw.head())
        except Exception as e:
            logger.exception("Data loading failed: %s", e)
            sys.exit(1)

    # Data validation
    if args.stage in ["validate", "all"]:
        schema = config.get("data_validation", {}).get("schema", {}).get("columns", [])
        try:
            validate_schema(df_raw, schema)
            logger.info("Data validation passed.")
        except Exception as e:
            logger.exception("Data validation failed: %s", e)
            sys.exit(1)

    # Run the full ML pipeline (feature engineering, preprocessing, modeling)
    if args.stage in ["pipeline", "all"]:
        logger.info("Starting full ML pipeline (preprocessing, feature engineering, modeling)...")
        try:
            run_model_pipeline(df_raw, config)
            logger.info("ML pipeline completed successfully.")
        except Exception as e:
            logger.exception("ML pipeline failed: %s", e)
            sys.exit(1)

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
