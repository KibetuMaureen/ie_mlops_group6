"""
preprocessing.py

This module contains preprocessing utilities for structured datasets.
It provides functions to:
- Apply label encoding to selected columns.
- Construct a scikit-learn preprocessing pipeline.
- Retrieve feature names after transformation.

The script can also be run as a standalone module to preprocess data
using a config YAML file.

Usage (as script):
    python -m src.preprocessing.preprocessing input.csv output.csv config.yaml
"""

import logging
import sys

from pathlib import Path

import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)

logger = logging.getLogger(__name__)


def preprocess_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply label encoding to categorical columns specified in the config.
    This function is for standalone scripts, not the main pipeline.
    """
    try:
        logger.info("Starting preprocessing...")
        df = df.copy()
        label_encode_cols = config.get("preprocessing", {}).get(
            "label_encode", []
        )
        # Label Encoding (replace original column)
        label_encoders = {}
        for col in label_encode_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le
        logger.info(f"Label encoding applied to: {label_encode_cols}")
        logger.info("Preprocessing completed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

def build_preprocessing_pipeline(config: dict) -> Pipeline:
    """
    Construct an sklearn preprocessing pipeline for numeric and categorical features.
    Numeric: impute missing values with mean, then scale.
    Categorical: impute missing with most frequent, then one-hot encode.
    """
    numeric_features = config.get("preprocessing", {}).get("numeric", [])
    categorical_features = config.get("preprocessing", {}).get("categorical", [])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor)
    ])
    return pipeline

def get_output_feature_names(
    preprocessor: Pipeline, input_features: list, config: dict
) -> list:
    """
    Retrieve the list of output feature names after transformation.
    Handles OneHotEncoder output for categorical columns.
    """
    col_transform = preprocessor.named_steps["preprocessor"]
    feature_names = []
    # Numeric features
    numeric_features = config.get("preprocessing", {}).get("numeric", [])
    feature_names.extend(numeric_features)
    # Categorical features (OneHotEncoder)
    categorical_features = config.get("preprocessing", {}).get("categorical", [])
    for name, trans, cols in col_transform.transformers_:
        if name == "cat" and hasattr(trans.named_steps["encoder"], "get_feature_names_out"):
            cats = trans.named_steps["encoder"].get_feature_names_out(cols)
            feature_names.extend(cats)
    return feature_names

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    if len(sys.argv) < 4:
        logger.error(
            "Usage: python -m src.preprocessing.preprocessing "
            "<input.csv> <output.csv> <config.yaml>"
        )
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    config_path = sys.argv[3]
    try:
        df = pd.read_csv(input_path)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info("Data loaded successfully. Shape: %s", df.shape)
        df_processed = preprocess_data(df, config)
        df_processed.to_csv(output_path, index=False)
        logger.info("Preprocessed data saved to %s", output_path)
    except Exception as e:
        logger.exception("Preprocessing failed: %s", e)
        sys.exit(1)
