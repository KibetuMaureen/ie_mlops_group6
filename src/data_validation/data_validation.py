import pandas as pd
import logging
import sys
from typing import Any, List, Dict


class DataValidationError(Exception):
    """Raised when validation fails."""
    pass


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
                    f"Column '{name}' has {missing_count} missing values (required)."
                )
                raise DataValidationError(
                    f"Column '{name}' has {missing_count} missing values (required)."
                )
            else:
                logging.warning(
                    f"Column '{name}' has {missing_count} missing values (optional)."
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


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="Validate a CSV file against defined schema in config.yaml"
    )
    parser.add_argument(
        "data_csv",
        type=str,
        help="Path to the CSV data file to validate."
    )
    parser.add_argument(
        "config_yaml",
        type=str,
        help="Path to the YAML config file containing the schema."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    try:
        df = pd.read_csv(args.data_csv)
    except Exception as e:
        logging.error(f"Failed to load data file: {e}")
        sys.exit(1)

    try:
        with open(args.config_yaml, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load config file: {e}")
        sys.exit(1)

    schema = config.get("data_validation", {}).get("schema", {}).get("columns", [])
    try:
        validate_schema(df, schema)
        logging.info("Data validation completed successfully.")
    except DataValidationError as e:
        logging.error(f"Data validation failed: {e}")
        sys.exit(1)
