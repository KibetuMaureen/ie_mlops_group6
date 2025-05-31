import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def preprocess_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply label encoding to categorical columns specified in the config.

    This function replaces original categorical columns with label-encoded
    versions.

    Args:
        df (pd.DataFrame): Input dataframe to preprocess.
        config (dict):

        Configuration dictionary containing preprocessing parameters.

    Returns:
        pd.DataFrame: DataFrame with label-encoded columns.
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
    Construct an sklearn preprocessing pipeline for numeric features.

    The pipeline imputes missing values with the mean and scales features
    using StandardScaler. Other feature types are passed through unchanged.

    Args:
        config (dict): Configuration dictionary with preprocessing details.

    Returns:
        Pipeline: sklearn Pipeline object for preprocessing.
    """
    numeric_features = config.get("preprocessing", {}).get("numeric", [])

    # Numeric pipeline: impute missing values, then scale
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Combine into a ColumnTransformer
    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough"
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

    This extracts feature names from the numeric features in the config and
    any passthrough columns retained by the ColumnTransformer.

    Args:
        preprocessor (Pipeline): The preprocessing pipeline.
        input_features (list): List of input feature names.
        config (dict): Configuration dictionary containing feature info.

    Returns:
        list: List of transformed feature names.
    """
    col_transform = preprocessor.named_steps["preprocessor"]
    feature_names = []

    # Numeric features
    numeric_features = config.get("preprocessing", {}).get("numeric", [])
    feature_names.extend(numeric_features)

    # Add passthrough features (if any)
    passthrough = (
        col_transform.transformers_[-1][2]
        if col_transform.transformers_[-1][0] == "remainder"
        else []
    )
    feature_names.extend(passthrough)

    return feature_names


if __name__ == "__main__":
    import sys
    import yaml
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
