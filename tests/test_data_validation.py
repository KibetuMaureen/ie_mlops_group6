import pytest
import os
import pandas as pd
import numpy as np
import yaml
import logging
from src.data_validation.data_validation import validate_schema, DataValidationError


@pytest.fixture(scope="module")
def schema():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["data_validation"]["schema"]["columns"]


@pytest.fixture
def valid_df():
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


def test_missing_required_column_raises(valid_df, schema):
    # Remove a required column from df and schema requires it
    schema_req = [dict(col) for col in schema]
    # Mark a column required, e.g. 'cc_num'
    for col in schema_req:
        if col["name"] == "cc_num":
            col["required"] = True

    df = valid_df.copy()
    df.drop(columns=["cc_num"], inplace=True)

    with pytest.raises(DataValidationError) as e:
        validate_schema(df, schema_req)
    assert "Missing required column: cc_num" in str(e.value)


def test_missing_values_in_required_column_raises(valid_df, schema):
    schema_req = [dict(col) for col in schema]
    for col in schema_req:
        if col["name"] == "amt":
            col["required"] = True

    df = valid_df.copy()
    df.loc[0, "amt"] = None

    with pytest.raises(DataValidationError) as e:
        validate_schema(df, schema_req)
    assert "has 1 missing values (required)" in str(e.value)


@pytest.mark.parametrize("dtype, valid_val, invalid_val", [
    ("int", 10, "string"),
    ("float", 10.0, "string"),
    ("str", "text", 123),
])
def test_dtype_checking(valid_df, schema, dtype, valid_val, invalid_val):
    # Add test column with dtype check
    df = valid_df.copy()
    df["test_col"] = [invalid_val]
    schema_test = schema + [{"name": "test_col", "dtype": dtype, "required": True}]

    with pytest.raises(DataValidationError) as e:
        validate_schema(df, schema_test)
    assert "incorrect type" in str(e.value)

    # Test with valid value does not raise
    df["test_col"] = [valid_val]
    validate_schema(df, schema_test)


def test_dtype_allows_nan(valid_df, schema):
    # Nan should be allowed and not cause type error
    df = valid_df.copy()
    df["test_col"] = [np.nan]
    schema_test = schema + [{"name": "test_col", "dtype": "int", "required": False}]
    validate_schema(df, schema_test)  # Should not raise


def test_allowed_values_multiple(valid_df, schema):
    df = valid_df.copy()
    df = pd.concat([df]*3, ignore_index=True)
    df.loc[1, "gender"] = "F"
    df.loc[2, "gender"] = "X"  # invalid

    schema_allowed = []
    for col in schema:
        c = col.copy()
        if c["name"] == "gender":
            c["allowed_values"] = ["M", "F"]
        schema_allowed.append(c)

    with pytest.raises(DataValidationError) as e:
        validate_schema(df, schema_allowed)
    assert "outside of allowed set" in str(e.value)


def test_min_value_boundary(valid_df, schema):
    df = valid_df.copy()
    schema_min = []
    for col in schema:
        c = col.copy()
        if c["name"] == "amt":
            c["min"] = 0
        schema_min.append(c)

    # amt exactly at min (should pass)
    df["amt"] = 0.0
    df["amt"] = df["amt"].astype(float)
    validate_schema(df, schema_min)

    # amt just below min (should raise)
    df["amt"] = -0.01
    df["amt"] = df["amt"].astype(float)
    with pytest.raises(DataValidationError):
        validate_schema(df, schema_min)


def test_empty_schema_allows_any_df(valid_df):
    # Empty schema means no validation, should pass
    validate_schema(valid_df, [])


def test_empty_dataframe_with_schema_raises_if_required(schema):
    # Empty dataframe with required columns should raise
    schema_req = [dict(col) for col in schema]
    for col in schema_req:
        col["required"] = True

    df = pd.DataFrame()
    with pytest.raises(DataValidationError) as e:
        validate_schema(df, schema_req)
    # Expect missing required column error (first required column)
    assert "Missing required column" in str(e.value)


def test_multiple_issues_only_first_raises(valid_df, schema):
    df = valid_df.copy()
    df.drop(columns=["cc_num"], inplace=True)  # Missing required
    # Mark cc_num required
    schema_req = [dict(col) for col in schema]
    for col in schema_req:
        if col["name"] == "cc_num":
            col["required"] = True
        if col["name"] == "amt":
            col["min"] = 0

    # amt also invalid but expect missing column error first
    df["amt"] = -10

    with pytest.raises(DataValidationError) as e:
        validate_schema(df, schema_req)
    assert "Missing required column: cc_num" in str(e.value)
