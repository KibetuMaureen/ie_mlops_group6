import pandas as pd
import pytest
from unittest.mock import patch
import subprocess
import yaml
from src.features.features import calculate_distance, add_engineered_features


def test_calculate_distance_valid():
    """
    Test the `calculate_distance` function with valid input data.

    This test verifies that the `calculate_distance` function correctly
    calculates a positive distance when provided with a dictionary containing
    valid latitude and longitude values for both the origin and the
    destination.

    Test Data:
    - Origin: Latitude 40.7128, Longitude -74.0060 (New York City)
    - Destination: Latitude 34.0522, Longitude -118.2437 (Los Angeles)

    Assertions:
    - The calculated distance should be greater than 0.
    """
    row = {
        "lat": 40.7128,
        "long": -74.0060,
        "merch_lat": 34.0522,
        "merch_long": -118.2437
    }
    distance = calculate_distance(row)
    assert distance > 0


def test_calculate_distance_exception_handling(caplog):
    """
    Test the `calculate_distance` function's exception handling.

    Verifies that the function returns 0.0 and logs a warning when an
    exception occurs during distance calculation.

    Test Data:
    - Mocked geodesic function to raise a ValueError.

    Assertions:
    - The returned distance should be 0.0.
    - The warning message should contain specific error details.
    """
    row = {
        "lat": 40.7128,
        "long": -74.0060,
        "merch_lat": 34.0522,
        "merch_long": -118.2437
    }

    with patch(
        "src.features.features.geodesic",
        side_effect=ValueError("Mocked error")
    ), caplog.at_level("WARNING"):
        distance = calculate_distance(row)
        assert distance == 0.0
        assert "Geo distance calculation failed" in caplog.text
        assert "Mocked error" in caplog.text


def test_add_engineered_features(caplog):
    """
    Test the `add_engineered_features` function with valid input data.

    Verifies that the function correctly adds new features to the DataFrame,
    including datetime-based features, age, and geo distance.

    Test Data:
    - A single-row DataFrame with valid transaction and user data.

    Assertions:
    - The resulting DataFrame should contain all expected feature columns.
    - The values of the new features should match expected results.
    """
    df = pd.DataFrame([{
        "trans_date_trans_time": "2022-01-15 14:30:00",
        "dob": "1990-01-01",
        "lat": 40.7128,
        "long": -74.0060,
        "merch_lat": 34.0522,
        "merch_long": -118.2437
    }])

    with caplog.at_level("INFO"):
        result = add_engineered_features(df)

    expected_columns = [
        "hour", "day", "month", "day_of_week",
        "is_weekend", "age", "geo_distance"
    ]

    for col in expected_columns:
        assert col in result.columns

    assert result["hour"].iloc[0] == 14
    assert result["is_weekend"].iloc[0] == 1  # Jan 15, 2022 was a Saturday
    assert result["age"].iloc[0] == 32
    assert result["geo_distance"].iloc[0] > 0
    assert "Computed geo distance" in caplog.text
    assert "Feature engineering complete" in caplog.text


def test_add_engineered_features_missing_columns():
    """
    Test the `add_engineered_features` function with missing columns.

    Verifies that the function raises an appropriate error when required
    columns are missing from the input DataFrame.

    Test Data:
    - A DataFrame missing the `trans_date_trans_time` column.

    Assertions:
    - The function should raise a KeyError.
    """
    df = pd.DataFrame([{
        "dob": "1990-01-01",
        "lat": 40.7128,
        "long": -74.0060,
        "merch_lat": 34.0522,
        "merch_long": -118.2437
    }])

    try:
        add_engineered_features(df)
    except KeyError as e:
        assert "trans_date_trans_time" in str(e)


def test_add_engineered_features_invalid_datetime():
    """
    Test the `add_engineered_features` function with invalid datetime values.

    Verifies that the function raises an error when the `trans_date_trans_time`
    column contains invalid datetime strings.

    Test Data:
    - A DataFrame with an invalid datetime string in the
    `trans_date_trans_time` column.

    Assertions:
    - The function should raise a ValueError.
    """
    df = pd.DataFrame([{
        "trans_date_trans_time": "invalid_datetime",  # this will fail
        "dob": "1990-01-01",
        "lat": 40.7128,
        "long": -74.0060,
        "merch_lat": 34.0522,
        "merch_long": -118.2437
    }])

    with pytest.raises(ValueError) as excinfo:
        add_engineered_features(df)

    assert "unable to parse" in str(excinfo.value)


def test_features_cli(tmp_path):
    """
    Test the CLI functionality of the features module.

    Verifies that the script processes input data and generates a valid
    output file with engineered features.

    Test Data:
    - A valid input CSV file and a corresponding config.yaml file.

    Assertions:
    - The script should run successfully and produce an output file.
    - The output file should contain the expected feature columns.
    """
    input_data = pd.DataFrame([{
        "trans_date_trans_time": "2022-01-15 14:30:00",
        "dob": "1990-01-01",
        "lat": 40.7128,
        "long": -74.0060,
        "merch_lat": 34.0522,
        "merch_long": -118.2437
    }])
    input_csv = tmp_path / "raw.csv"
    output_csv = tmp_path / "features.csv"
    input_data.to_csv(input_csv, index=False)

    config = {
        "data_source": {
            "raw_path": str(input_csv),
            "features_path": str(output_csv)
        }
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = subprocess.run(
        ["python", "-m", "src.features.features", str(config_path)],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert output_csv.exists()
    df = pd.read_csv(output_csv)
    assert "geo_distance" in df.columns


def test_cli_invalid_lat_long(tmp_path):
    """
    Test the CLI functionality with invalid latitude/longitude values.

    Verifies that the script handles invalid geo-coordinates gracefully
    and logs appropriate warnings.

    Test Data:
    - A CSV file with invalid latitude and longitude values.

    Assertions:
    - The script should complete execution without crashing.
    - The geo_distance column should contain 0.0 for invalid rows.
    """
    input_data = pd.DataFrame([{
        "trans_date_trans_time": "2022-01-15 14:30:00",
        "dob": "1990-01-01",
        "lat": "invalid_lat",
        "long": "invalid_long",
        "merch_lat": 34.0522,
        "merch_long": -118.2437
    }])
    input_csv = tmp_path / "raw.csv"
    output_csv = tmp_path / "features.csv"
    input_data.to_csv(input_csv, index=False)

    config = {
        "data_source": {
            "raw_path": str(input_csv),
            "features_path": str(output_csv)
        }
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = subprocess.run(
        ["python", "-m", "src.features.features", str(config_path)],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert output_csv.exists()
    df = pd.read_csv(output_csv)
    assert "geo_distance" in df.columns
    assert df["geo_distance"].iloc[0] == 0.0


def test_cli_missing_arguments():
    """
    Test case for verifying the behavior of the CLI when required arguments
    are missing.

    This test runs the `src.features.features` module as a script using
    `subprocess.run`without providing the necessary arguments. It asserts
    that the process exits with a return code of 1 and that the error
    message in `stderr` contains the expected usage
    instructions.
    """
    result = subprocess.run(
        ["python", "-m", "src.features.features"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 1
    assert "Usage: python -m src.features.features" in result.stderr


def test_cli_invalid_config(tmp_path):
    config_path = tmp_path / "invalid_config.yaml"
    config_path.write_text("not: valid: yaml: [")  # malformed

    result = subprocess.run(
        ["python", "-m", "src.features.features", str(config_path)],
        capture_output=True,
        text=True
    )
    assert result.returncode == 1
    assert "Failed to load config or paths" in result.stderr


def test_cli_missing_csv(tmp_path):

    config_path = tmp_path / "config.yaml"
    config = {
        "data_source": {
            "raw_path": str(tmp_path / "nonexistent.csv"),
            "features_path": str(tmp_path / "output.csv")
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = subprocess.run(
        ["python", "-m", "src.features.features", str(config_path)],
        capture_output=True,
        text=True
    )
    assert result.returncode == 1
    assert "Failed to load data" in result.stderr


def test_cli_feature_engineering_failure(tmp_path):

    bad_df_path = tmp_path / "bad_input.csv"
    bad_df_path.write_text("bad_column\\n123\\n456")  # missing required fields

    config = {
        "data_source": {
            "raw_path": str(bad_df_path),
            "features_path": str(tmp_path / "output.csv")
        }
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = subprocess.run(
        ["python", "-m", "src.features.features", str(config_path)],
        capture_output=True,
        text=True
    )
    assert result.returncode == 1
    assert "Feature engineering failed" in result.stderr