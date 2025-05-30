"""
This module contains unit tests for the preprocessing functionality.

The tests cover:
- Data preprocessing with label encoding.
- Building and validating preprocessing pipelines.
- Extracting output feature names from pipelines.
- Testing the preprocessing CLI script.
"""

import subprocess
from pathlib import Path
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from src.preprocessing.preprocessing import (
    preprocess_data,
    build_preprocessing_pipeline,
    get_output_feature_names,
)


def test_preprocess_data_label_encoding():
    """
    Test the preprocess_data function to ensure it correctly applies label
    encoding to the specified categorical columns in the input DataFrame.
    """
    df = pd.DataFrame({
        "category": ["A", "B", "A", "C"],
        "value": [1, 2, 3, 4]
    })
    config = {"preprocessing": {"label_encode": ["category"]}}
    result = preprocess_data(df, config)

    assert "category" in result.columns
    assert result["category"].isin([0, 1, 2]).all()


def test_build_preprocessing_pipeline_transform():
    """
    Test the `build_preprocessing_pipeline` function to ensure it creates a
    valid preprocessing pipeline.

    This test verifies:
    1. The returned object is an instance of `Pipeline`.
    2. The pipeline correctly processes the input DataFrame by:
        - Handling numeric columns specified in the configuration.
        - Passing through non-numeric columns.
    3. The output shape matches the expected dimensions after transformation.

    Test Data:
    - Input DataFrame contains one numeric column with missing values and one
    passthrough column.
    - Configuration specifies preprocessing for the numeric column.

    Assertions:
    - The pipeline is an instance of `Pipeline`.
    - The transformed DataFrame has the correct shape (4 rows, 2 columns).
    """
    df = pd.DataFrame({
        "num1": [1, 2, None, 4],
        "pass": ["a", "b", "c", "d"]
    })
    config = {"preprocessing": {"numeric": ["num1"]}}
    pipeline = build_preprocessing_pipeline(config)

    assert isinstance(pipeline, Pipeline)

    # Fit-transform and verify output shape
    result = pipeline.fit_transform(df)
    assert result.shape == (4, 2)  # 1 numeric + 1 passthrough


def test_get_output_feature_names():
    """
    Test case for the `get_output_feature_names` function.

    This test verifies that the `get_output_feature_names` function correctly
    returns the output feature names after applying a preprocessing pipeline
    to a DataFrame. The pipeline is built based on a configuration dictionary
    and is expected to handle numeric and passthrough features.

    Test Steps:
    1. Create a sample DataFrame with numeric and text columns.
    2. Define a configuration dictionary specifying preprocessing for numeric
    columns.
    3. Build and fit a preprocessing pipeline using the configuration.
    4. Call `get_output_feature_names` with the pipeline, original column
    names
       and configuration.
    5. Assert that the output feature names match the expected list, which
    includes
       the numeric column name and the index for the passthrough column.

    Expected Output:
    The output feature names should be `["num1", 1]`, where "num1" is the
    numeric
    column and `1` represents the index of the passthrough column.
    """
    df = pd.DataFrame({
        "num1": [1, 2, 3, 4],
        "text": ["x", "y", "z", "w"]
    })
    config = {"preprocessing": {"numeric": ["num1"]}}
    pipeline = build_preprocessing_pipeline(config)
    pipeline.fit(df)

    output_features = get_output_feature_names(pipeline, list(df.columns),
                                               config)

    # Expecting index 1 because passthrough uses column index
    assert output_features == ["num1", 1]


def test_get_output_feature_names_else_branch():
    """
    Test the `get_output_feature_names` function for the "else" branch
    scenario.

    This test verifies that the function correctly extracts the output feature
    names when the preprocessor's transformer contains a specific
    configuration. It uses a dummy preprocessor and transformer to simulate
    the behavior of the function.

    The test checks:
    - That the function correctly identifies and returns the numeric features
      specified in the configuration.

    Assertions:
    - Ensures that the output feature names match the expected list of numeric
    features.
    """

    class DummyTransformer:
        """
        A dummy transformer class used for testing purposes.

        Attributes:
            transformers_ (list): A list of tuples representingtransformers.
                                Each tuple contains a name, a transformer
                                object (or None),
                                and a list of feature names.
        """
        transformers_ = [("num", None, ["a"])]

    class DummyPreprocessor:
        """
        A dummy preprocessor class used for testing purposes.

        Attributes:
            named_steps (dict): A dictionary containing the steps of the
                                preprocessing pipeline. In this case, it
                                includes a "preprocessor" step with a
                                DummyTransformer instance.
        """
        named_steps = {"preprocessor": DummyTransformer()}

    config = {"preprocessing": {"numeric": ["a"]}}
    features = get_output_feature_names(DummyPreprocessor(), ["a", "b"],
                                        config)
    assert features == ["a"]


def test_preprocessing_cli(tmp_path: Path):
    """
    Test the preprocessing CLI script.

    This test verifies that the preprocessing script correctly processes an
    input CSV fileaccording to a given configuration and produces the expected
    output CSV file.

    Steps:
    1. Create a temporary input CSV file with sample data.
    2. Create a temporary configuration YAML file specifying preprocessing
    steps:
       - Label encoding for categorical columns.
       - Numeric processing for numerical columns.
    3. Run the preprocessing script using subprocess with the input file,
    output file, and configuration file as arguments.
    4. Verify that the script runs successfully (exit code 0).
    5. Check that the output file is created.
    6. Validate the contents of the output file to ensure the expected columns
    are present.

    Args:
        tmp_path (pathlib.Path): Temporary directory provided by pytest for
        creating temporary files and directories.
    """

    # Create a temporary CSV file
    df = pd.DataFrame([{
        "category": "A",
        "num1": 1.0
    }])
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    config_path = tmp_path / "config.yaml"

    df.to_csv(input_path, index=False)

    config = {
        "preprocessing": {
            "label_encode": ["category"],
            "numeric": ["num1"]
        }
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    # Run the script using subprocess and capture output
    result = subprocess.run(
        ["python", "-m", "src.preprocessing.preprocessing", str(input_path),
         str(output_path), str(config_path)],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert output_path.exists()

    df_out = pd.read_csv(output_path)
    assert "category" in df_out.columns
    assert "num1" in df_out.columns


def test_cli_missing_arguments():
    """
    Test case for verifying the behavior of the CLI when required arguments
    are missing.

    This test executes the preprocessing module as a script using subprocess
    and checks:
    1. That the return code is 1, indicating an error due to missing
    arguments.
    2. That the error message in stderr contains the expected usage
    instructions.

    Assertions:
    - The return code of the subprocess should be 1.
    - The standard error output should include the expected usage message.
    """

    result = subprocess.run(
        ["python", "-m", "src.preprocessing.preprocessing"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 1
    assert "Usage: python -m src.preprocessing.preprocessing" in result.stderr
