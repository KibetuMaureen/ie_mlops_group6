import pandas as pd
import logging
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

        if required and name not in df.columns:
            raise DataValidationError(f"Missing required column: {name}")

        if name not in df.columns:
            continue

        if dtype:
            expected_type = {"int": int, "float": float, "str": str}.get(dtype)
            if expected_type:
                if not df[name].map(lambda x: isinstance(x, expected_type) or pd.isnull(x)).all():
                    raise DataValidationError(f"Column '{name}' has incorrect type. Expected {dtype}.")

        if allowed:
            if not df[name].isin(allowed).all():
                raise DataValidationError(f"Column '{name}' contains values outside of allowed set.")

        if min_val is not None:
            if (df[name] < min_val).any():
                raise DataValidationError(f"Column '{name}' has values below minimum of {min_val}.")

    logging.info("Validator: Schema validation passed.")
