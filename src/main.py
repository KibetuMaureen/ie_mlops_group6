"""
main.py

Minimal entry point to check only data loading.
"""

import argparse
import sys
import logging
import os
import yaml
from src.data_loader.data_loader import get_data

logger = logging.getLogger(__name__)


def setup_logging():
    """
    Set up simple logging to console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load the configuration YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        dict: Loaded configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """
    Main function to parse arguments and check data loading.
    """
    parser = argparse.ArgumentParser(
        description="Minimal data loading checker"
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
        help="Path to an optional .env file for secrets or environment variables"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="load",
        choices=["load"],
        help="Pipeline stage to execute (only 'load' is supported in this script)"
    )
    args = parser.parse_args()

    setup_logging()

    if args.stage == "load":
        # Load configuration
        try:
            _ = load_config(args.config)
        except Exception as e:
            logger.exception("Failed to load config: %s", e)
            sys.exit(1)

        # Load data
        try:
            df_raw = get_data(
                config_path=args.config,
                env_path=args.env,
                data_stage="raw"
            )
            if df_raw is None or not hasattr(df_raw, "shape"):
                logger.error(
                    "Data loading failed: get_data did not return a valid DataFrame"
                )
                sys.exit(1)
            logger.info(
                "Raw data loaded successfully. Shape: %s",
                df_raw.shape
            )
            # Show first few rows for inspection
            print(df_raw.head())
        except Exception as e:
            logger.exception("Data loading failed: %s", e)
            sys.exit(1)

        logger.info("Data loading check completed successfully.")


if __name__ == "__main__":
    main()
