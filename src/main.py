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
    log_file = logging_config.get("log_file", "logs/main.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
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
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="ML pipeline entry point with staging support"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config YAML file")
    parser.add_argument("--env", type=str, default=".env", help="Optional .env path")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["load", "validate", "pipeline", "infer", "all"],
                        help="Pipeline stage to execute")

    args = parser.parse_args()

    try:
        config = load_config(args.config)
        setup_logging(config.get("logging", {}))
    except Exception as e:
        logger.exception("Startup failed: %s", e)
        sys.exit(1)

    logger.info("Pipeline started")
    df_raw = None

    if args.stage in ["load", "validate", "pipeline", "all"]:
        try:
            df_raw = get_data(config_path=args.config, env_path=args.env, data_stage="raw")
            logger.info("Raw data loaded. Shape: %s", df_raw.shape)
            print(df_raw.head())
        except Exception as e:
            logger.exception("Data loading failed: %s", e)
            sys.exit(1)

    if args.stage in ["validate", "all"]:
        try:
            schema = config.get("data_validation", {}).get("schema", {}).get("columns", [])
            validate_schema(df_raw, schema)
            logger.info("Data validation passed.")
        except Exception as e:
            logger.exception("Data validation failed: %s", e)
            sys.exit(1)

    if args.stage in ["pipeline", "all"]:
        logger.info("Starting full ML pipeline...")
        try:
            run_model_pipeline(df_raw, config)
            logger.info("ML pipeline completed.")
        except Exception as e:
            logger.exception("ML pipeline failed: %s", e)
            sys.exit(1)

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
