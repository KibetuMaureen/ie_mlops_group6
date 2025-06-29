"""
data_validation/run.py

MLflow-compatible
Modular data validation step with Hydra config
W&B logging
"""

import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import hydra
import pandas as pd
import yaml
from dotenv import load_dotenv
from omegaconf import DictConfig

import wandb


# Import the validation functions from data_validation.py
from data_validation import (
    validate_schema,
    handle_missing_values,
    DataValidationError
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("data_validation")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@hydra.main(config_path=str(PROJECT_ROOT), config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the data validation pipeline.

    This function performs the following steps:
    1. Initializes a Weights & Biases (W&B) run for experiment tracking.
    2. Loads the latest raw data artifact from W&B.
    3. Validates the data schema using rules from a YAML config file.
    4. Handles missing values if validation passes.
    5. Logs the validated dataset back to W&B as an artifact.
    6. Updates W&B summary metrics based on validation results.
    7. Handles and logs any validation errors or unexpected exceptions.

    Args:
        cfg (DictConfig): Hydra config object with parameters from config file
    """
    config_path = PROJECT_ROOT / "config.yaml"

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"data_validation_{dt_str}"

    run = None
    try:
        run = wandb.init(
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
            job_type="data_validation",
            name=run_name,
            config=dict(cfg),
            tags=["data_validation"]
        )
        logger.info("Started WandB run: %s", run_name)

        # Load raw data artifact from W&B
        raw_art = run.use_artifact("raw_data:latest")
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = raw_art.download(root=tmp_dir)
            df = pd.read_csv(os.path.join(raw_path, "raw_data.csv"))
        if df.empty:
            logger.warning("Loaded dataframe is empty.")

        # Load config dict for validation
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        schema = config_dict.get("data_validation", {}) \
            .get("schema", {}) \
            .get("columns", [])

        # Validate schema
        try:
            validate_schema(df, schema)
            logger.info("Data validation completed successfully.")
            df = handle_missing_values(df)
        except DataValidationError as e:
            logger.error("Data validation failed: %s", e)
            wandb.summary["validation_result"] = "failed"
            wandb.summary["validation_error"] = str(e)
            sys.exit(1)

        # Save validated data to a temporary CSV and log to W&B
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "validated_data.csv"
            df.to_csv(tmp_path, index=False)
            val_artifact = wandb.Artifact("validated_data", type="dataset")
            val_artifact.add_file(str(tmp_path), name="validated_data.csv")
            run.log_artifact(val_artifact, aliases=["latest"])
            logger.info("Logged validated data artifact to WandB")

        # Log validation summary to W&B
        wandb.summary.update({
            "validation_result": "passed",
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "columns": list(df.columns)
        })

    except Exception as e:  # pylint: disable=broad-except
        logger.exception("Failed during data validation step")
        if run is not None:
            run.alert(title="Data Validation Error", text=str(e))
        sys.exit(1)
    finally:
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
