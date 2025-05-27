import pytest
import os
import pandas as pd
import yaml
from src.data_validation.data_validation import validate_schema, DataValidationError

# Load schema from config.yaml
config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

schema = config["data_validation"]["schema"]["columns"]


def get_valid_df():
    return pd.DataFrame({
        "trans_date_trans_time": ["2020-06-01"],
        "cc_num": [1234567890],
        "merchant": ["amazon"],
        "category": ["grocery_pos"],
        "amt": [120.0],
        "first": ["John"],
        "last": ["Doe"],
        "gender": ["M"],
        "street": ["123 Main St"],
        "city": ["San Francisco"],
        "state": ["CA"],
        "zip": [94105],
        "lat": [37.7749],
        "long": [-122.4194],
        "city_pop": [870000],
        "job": ["Engineer"],
        "dob": ["1985-01-01"],
        "trans_num": ["abcd1234"],
        "unix_time": [1371922345],
        "merch_lat": [37.8],
        "merch_long": [-122.4],
        "is_fraud": [0]
    })


def test_column_missing_but_not_required():
    df = get_valid_df()
    schema_copy = schema + [{"name": "optional_column", "dtype": "str", "required": False}]
    validate_schema(df, schema_copy)


def test_missing_optional_column_logs_warning(caplog):
    df = get_valid_df()
    schema_copy = schema + [{"name": "optional_column", "dtype": "str", "required": False}]
    caplog.set_level("WARNING")
    validate_schema(df, schema_copy)
    assert "not found in DataFrame" in caplog.text


def test_missing_value_in_optional_column_logs_warning(caplog):
    df = get_valid_df()
    df["optional_column"] = [None]
    schema_copy = schema + [{"name": "optional_column", "dtype": "str", "required": False}]
    caplog.set_level("WARNING")
    validate_schema(df, schema_copy)
    assert "missing values (optional)" in caplog.text


def test_logging_of_type_mismatch(caplog):
    df = get_valid_df()
    df["cc_num"] = ["wrong"]
    caplog.set_level("ERROR")
    with pytest.raises(DataValidationError):
        validate_schema(df, schema)
    assert "incorrect type" in caplog.text


def test_logging_of_allowed_values_violation(caplog):
    df = get_valid_df()
    df["gender"] = ["X"]
    schema_with_allowed = []
    for col in schema:
        if col["name"] == "gender":
            col = col.copy()
            col["allowed_values"] = ["M", "F"]
        schema_with_allowed.append(col)
    caplog.set_level("ERROR")
    with pytest.raises(DataValidationError):
        validate_schema(df, schema_with_allowed)
    assert "outside of allowed set" in caplog.text


def test_logging_of_min_value_violation(caplog):
    df = get_valid_df()
    df["amt"] = [-100.0]
    schema_with_min = []
    for col in schema:
        if col["name"] == "amt":
            col = col.copy()
            col["min"] = 0
        schema_with_min.append(col)
    caplog.set_level("ERROR")
    with pytest.raises(DataValidationError):
        validate_schema(df, schema_with_min)
    assert "below minimum" in caplog.text


def test_unknown_dtype_is_ignored():
    df = get_valid_df()
    df["mystery"] = ["unknown"]
    schema_with_unknown = schema + [{"name": "mystery", "dtype": "mystery_type", "required": True}]
    validate_schema(df, schema_with_unknown)