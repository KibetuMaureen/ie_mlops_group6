import pytest
import os
import pandas as pd
import yaml
from src.data_validation.data_validation import validate_schema, DataValidationError

# Load schema from config.yaml once
config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

schema = config["data_validation"]["schema"]["columns"]


def test_valid_data_passes():
    df = pd.DataFrame({
        "trans_date_trans_time": ["2020-06-01", "2020-06-02"],
        "cc_num": [1234567890, 9876543210],
        "merchant": ["amazon", "walmart"],
        "category": ["grocery_pos", "shopping_net"],
        "amt": [120.0, 80.5],
        "first": ["John", "Alice"],
        "last": ["Doe", "Smith"],
        "gender": ["M", "F"],
        "street": ["123 Main St", "456 Oak Ave"],
        "city": ["San Francisco", "Austin"],
        "state": ["CA", "TX"],
        "zip": [94105, 73301],
        "lat": [37.7749, 30.2672],
        "long": [-122.4194, -97.7431],
        "city_pop": [870000, 950000],
        "job": ["Engineer", "Teacher"],
        "dob": ["1985-01-01", "1990-02-02"],
        "trans_num": ["abcd1234", "efgh5678"],
        "unix_time": [1371922345, 1371922378],
        "merch_lat": [37.8, 30.3],
        "merch_long": [-122.4, -97.7],
        "is_fraud": [0, 1]
    })

    validate_schema(df, schema)


def test_missing_required_column_fails():
    df = pd.DataFrame({
        "amt": [120.0],
        "gender": ["M"]
        # Missing other required columns
    })

    with pytest.raises(DataValidationError):
        validate_schema(df, schema)


def test_invalid_categorical_value_fails():
    df = pd.DataFrame({
        "amt": [120.0],
        "merch_lat": [37.1],
        "merch_long": [-121.4],
        "category": ["invalid_category"],
        "gender": ["F"],
        "state": ["CA"],
        "is_fraud": [0]
    })

    with pytest.raises(DataValidationError):
        validate_schema(df, schema)


def test_negative_amount_fails():
    df = pd.DataFrame({
        "amt": [-50.0],
        "merch_lat": [37.1],
        "merch_long": [-121.4],
        "category": ["grocery_pos"],
        "gender": ["F"],
        "state": ["CA"],
        "is_fraud": [0]
    })

    with pytest.raises(DataValidationError):
        validate_schema(df, schema)


def test_invalid_dtype_fails():
    df = pd.DataFrame({
        "trans_date_trans_time": ["2020-06-01"],
        "cc_num": ["not_a_number"],  # should be int
        "amt": [100.0],
        "gender": ["M"],
        "category": ["grocery_pos"],
        "state": ["CA"],
        "is_fraud": [0],
        "merch_lat": [37.8],
        "merch_long": [-122.4],
        "first": ["Jane"],
        "last": ["Doe"],
        "merchant": ["target"],
        "street": ["123 Fake St"],
        "city": ["New York"],
        "zip": [10001],
        "lat": [40.7],
        "long": [-74.0],
        "city_pop": [8500000],
        "job": ["Engineer"],
        "dob": ["1992-04-04"],
        "trans_num": ["tx001"],
        "unix_time": [1371922345]
    })

    with pytest.raises(DataValidationError):
        validate_schema(df, schema)
