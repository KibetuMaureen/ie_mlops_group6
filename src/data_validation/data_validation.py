import pandas as pd
import logging
import sys
from typing import Any, List, Dict
from hydra import main
from omegaconf import DictConfig

class DataValidationError(Exception):
    """Raised when validation fails."""
    pass


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using forward fill then backward fill.
    """
    df = df.copy()
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    logging.info("Missing values handled in data_validation.")
    return df


def validate_schema(df: pd.DataFrame, schema: List[Dict[str, Any]]) -> None:
    """
    Validate a DataFrame against schema definitions from the YAML config.

    Args:
        df (pd.DataFrame): Data to validate.
        schema (list of dict): Schema rules from config.yaml.

    Raises:
        DataValidationError: If validation checks fail.
    """
    logging.info("Validator: Starting schema validation...")

    for column in schema:
        name = column["name"]
        dtype = column.get("dtype")
        required = column.get("required", False)
        allowed = column.get("allowed_values")
        min_val = column.get("min")

        # Check for required columns
        if required and name not in df.columns:
            logging.error(f"Missing required column: {name}")
            raise DataValidationError(f"Missing required column: {name}")

        if name not in df.columns:
            logging.warning(
                f"Column '{name}' not found in DataFrame, skipping."
            )
            continue

        # Check for missing values
        missing_count = df[name].isnull().sum()
        if missing_count > 0:
            if required:
                logging.error(
                    "Column '%s' has %d missing values (required).",
                    name,
                    missing_count
                )
                raise DataValidationError(
                    f"Column '{name}' has {missing_count} missing values (required)."
                )
            else:
                logging.warning(
                    "Column '%s' has %d missing values (optional).",
                    name,
                    missing_count
                )

        # Check data type
        if dtype:
            expected_type = {"int": int, "float": float, "str": str}.get(dtype)
            if expected_type:
                type_check = df[name].map(
                    lambda x: isinstance(x, expected_type) or pd.isnull(x)
                ).all()
                if not type_check:
                    logging.error(
                        f"Column '{name}' has incorrect type. Expected {dtype}"
                    )
                    raise DataValidationError(
                        f"Column '{name}' has incorrect type. Expected {dtype}"
                    )

        # Check allowed values
        if allowed:
            invalid_mask = ~df[name].isin(allowed)
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                logging.error(
                    f"Column '{name}' contains {invalid_count} values "
                    f"outside of allowed set."
                )
                raise DataValidationError(
                    f"Column '{name}' contains values outside of allowed set."
                )

        # Check minimum value
        if min_val is not None:
            below_min = (df[name] < min_val).sum()
            if below_min > 0:
                logging.error(
                    f"Column '{name}' has {below_min} values below minimum "
                    f"of {min_val}."
                )
                raise DataValidationError(
                    f"Column '{name}' has values below minimum of {min_val}."
                )

    logging.info("Validator: Schema validation passed.")


@main(config_path="../../config", config_name="config", version_base=None)
def run_validation(cfg: DictConfig):
    """
    Hydra entry point for running the data validation pipeline.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    try:
        df = pd.read_csv(cfg.data_source.raw_path)
    except Exception as e:
        logging.error(f"Failed to load data file: {e}")
        sys.exit(1)

    schema = cfg.data_validation.schema.columns

    try:
        validate_schema(df, schema)
        logging.info("Data validation completed successfully.")
        df = handle_missing_values(df)

        if cfg.data_validation.get("output_path"):
            df.to_csv(cfg.data_validation.output_path, index=False)
            logging.info("Data with missing values handled saved to %s", cfg.data_validation.output_path)
            logging.info(f"File successfully saved to: {cfg.data_validation.output_path}")

    except DataValidationError as e:
        logging.error(f"Data validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error during missing value handling: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_validation()

