"""
main.py

Unified entry point for:
- Data loading
- Schema validation
- Model training pipeline
- Batch inference
"""

import argparse
import sys
import logging
import os
import yaml

from src.data_loader.data_loader import get_data
from src.data_validation.data_validation import validate_schema
from src.model.model import run_model_pipeline
from src.inferencer.inferencer import run_inference

logger = logging.getLogger(__name__)


def setup_logging(logging_config: dict):
    """
    Set up logging to both console and file using configuration from YAML.

    Args:
        logging_config (dict):

        Configuration dictionary with keys like 'log_file',
        'level', 'format', and 'datefmt'.
    """
    log_file = logging_config.get("log_file", "logs/main.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    log_format = logging_config.get(
        "format", "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    date_format = logging_config.get("datefmt", "%Y-%m-%d %H:%M:%S")
    log_level = getattr(
        logging,
        logging_config.get("level", "INFO").upper(),
        logging.INFO)

    # Remove all existing handlers to prevent duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create and configure file handler
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    # Create and configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))

    # Apply the handlers
    logging.basicConfig(
        level=log_level,
        handlers=[file_handler, console_handler]
        )


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load YAML configuration from a specified path.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file does not contain a dictionary.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Config file must be a YAML mapping (dict), got "
            f"{type(config).__name__}"
        )

    return config


def main():
    """
    Main function that parses arguments
    - executes specified ML pipeline stages.
    """
    parser = argparse.ArgumentParser(
        description="ML pipeline entry point with staging support"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config YAML file"
    )
    parser.add_argument(
        "--env",
        type=str,
        default=".env",
        help="Optional .env path"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["load", "validate", "pipeline", "infer", "all"],
        help="Pipeline stage to execute"
    )

    args = parser.parse_args()

    # Load configuration and initialize logging
    try:
        config = load_config(args.config)
        setup_logging(config.get("logging", {}))
    except Exception as e:
        logger.exception("Startup failed: %s", e)
        sys.exit(1)

    logger.info("Pipeline started")
    df_raw = None

    # Load data if needed
    if args.stage in ["load", "validate", "pipeline", "all"]:
        try:
            df_raw = get_data(
                config_path=args.config,
                env_path=args.env,
                data_stage="raw"
            )
            logger.info("Raw data loaded. Shape: %s", df_raw.shape)
            print(df_raw.head())
        except Exception as e:
            logger.exception("Data loading failed: %s", e)
            sys.exit(1)

    # Validate schema if needed
    if args.stage in ["validate", "all"]:
        try:
            schema = config.get("data_validation", {}) \
                .get("schema", {}) \
                .get("columns", [])
            validate_schema(df_raw, schema)
            logger.info("Data validation passed.")
        except Exception as e:
            logger.exception("Data validation failed: %s", e)
            sys.exit(1)

    # Run model training pipeline if needed
    if args.stage in ["pipeline", "all"]:
        logger.info("Starting full ML pipeline...")
        try:
            run_model_pipeline(df_raw, config)
            logger.info("ML pipeline completed.")
        except Exception as e:
            logger.exception("ML pipeline failed: %s", e)
            sys.exit(1)

    # Run batch inference if needed
    if args.stage == "infer":
        try:
            inference_cfg = config.get("inference", {})
            input_csv = inference_cfg["input_csv"]
            output_csv = inference_cfg["output_csv"]

            logger.info("Running batch inference...")
            run_inference(input_csv, args.config, output_csv)
            logger.info("Inference completed. Output saved to: %s", output_csv)
        except Exception as e:
            logger.exception("Inference failed: %s", e)
            sys.exit(1)

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
