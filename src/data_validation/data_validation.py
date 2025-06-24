"""Data validation utilities for fraud detection pipeline."""

import logging
import sys
from typing import Any, List, Dict
import pandas as pd


class DataValidationError(Exception):
    """Raised when validation fails."""
    # No need for pass statement


def handle_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using forward fill then backward fill.
    """
    df_copy = dataframe.copy()
    df_copy.fillna(method='ffill', inplace=True)
    df_copy.fillna(method='bfill', inplace=True)
    logging.info("Missing values handled in data_validation.")
    return df_copy


def validate_schema(
        dataframe: pd.DataFrame,
        schema_def: List[Dict[str, Any]]
) -> None:
    """
    Validate a DataFrame against schema definitions from the YAML config.

    Args:
        dataframe (pd.DataFrame): Data to validate.
        schema_def (list of dict): Schema rules from config.yaml.

    Raises:
        DataValidationError: If validation checks fail.
    """
    logging.info("Validator: Starting schema validation...")

    dtype_map = {"int": int, "float": float, "str": str}

    for column in schema_def:
        name = column["name"]
        dtype = column.get("dtype")
        required = column.get("required", False)
        allowed = column.get("allowed_values")
        min_val = column.get("min")

        # Check for required columns
        if required and name not in dataframe.columns:
            logging.error("Missing required column: %s", name)
            raise DataValidationError(f"Missing required column: {name}")

        if name not in dataframe.columns:
            logging.warning(
                "Column '%s' not found in DataFrame, skipping.",
                name
            )
            continue

        # Check for missing values
        missing_count = dataframe[name].isnull().sum()
        if missing_count > 0:
            if required:
                logging.error(
                    "Column '%s' has %d missing values (required).",
                    name,
                    missing_count
                )
                raise DataValidationError(
                    f"Column '{name}' has {missing_count} missing values"
                    " (required)."
                )
            else:
                logging.warning(
                    "Column '%s' has %d missing values (optional).",
                    name,
                    missing_count
                )

        # Check data type
        if dtype:
            expected_type = dtype_map.get(dtype)
            if expected_type:
                type_check = dataframe[name].map(
                    lambda x: isinstance(x, expected_type) or pd.isnull(x)
                ).all()
                if not type_check:
                    logging.error(
                        "Column '%s' has incorrect type. Expected %s",
                        name,
                        dtype
                    )
                    raise DataValidationError(
                        f"Column '{name}' has incorrect type. Expected {dtype}"
                    )

        # Check allowed values
        if allowed:
            invalid_mask = ~dataframe[name].isin(allowed)
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                logging.error(
                    "Column '%s' contains %d values outside of allowed set.",
                    name,
                    invalid_count
                )
                raise DataValidationError(
                    f"Column '{name}' contains values outside of allowed set."
                )

        # Check minimum value
        if min_val is not None:
            below_min = (dataframe[name] < min_val).sum()
            if below_min > 0:
                logging.error(
                    "Column '%s' has %d values below minimum of %s.",
                    name,
                    below_min,
                    min_val
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
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional path to save the data after handling missing values."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    try:
        dataframe = pd.read_csv(args.data_csv)
    except (OSError, pd.errors.ParserError) as e:
        logging.error("Failed to load data file: %s", e)
        sys.exit(1)

    try:
        with open(args.config_yaml, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        logging.error("Failed to load config file: %s", e)
        sys.exit(1)

    schema_columns = (
        config.get("data_validation", {})
        .get("schema", {})
        .get("columns", [])
    )
    try:
        validate_schema(dataframe, schema_columns)
        logging.info("Data validation completed successfully.")
        dataframe = handle_missing_values(dataframe)
        if args.output_csv:
            dataframe.to_csv(args.output_csv, index=False)
            logging.info(
                "Data with missing values handled saved to %s",
                args.output_csv
            )
    except DataValidationError as e:
        logging.error("Data validation failed: %s", e)
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-except
        logging.error("Error during missing value handling: %s", e)
        sys.exit(1)
