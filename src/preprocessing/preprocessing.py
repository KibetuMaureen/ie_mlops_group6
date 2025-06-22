import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import hydra
from omegaconf import DictConfig
import os
import joblib

logger = logging.getLogger(__name__)


def preprocess_data(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    try:
        logger.info("Starting preprocessing...")

        df = df.copy()
        label_encode_cols = config.preprocessing.label_encode

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


def build_preprocessing_pipeline(config: DictConfig) -> Pipeline:
    numeric_features = config.preprocessing.numeric

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

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


def get_output_feature_names(preprocessor: Pipeline, input_features: list, config: DictConfig) -> list:
    col_transform = preprocessor.named_steps["preprocessor"]
    feature_names = []

    numeric_features = config.preprocessing.numeric
    feature_names.extend(numeric_features)

    passthrough = (
        col_transform.transformers_[-1][2]
        if col_transform.transformers_[-1][0] == "remainder"
        else []
    )
    feature_names.extend(passthrough)

    return feature_names


@hydra.main(config_path='../../conf', config_name='config', version_base='1.3')
def main(cfg: DictConfig):
    logging.basicConfig(
        level=cfg.logging.level,
        format=cfg.logging.format,
        datefmt=cfg.logging.datefmt,
        filename=cfg.logging.log_file
    )

    logger.info("Hydra-preprocessing script started.")

    input_path = cfg.data_source.raw_path
    output_path = cfg.data_source.processed_path

    # Load data
    df = pd.read_csv(input_path)
    logger.info("Data loaded successfully. Shape: %s", df.shape)

    # Preprocess
    df_processed = preprocess_data(df, cfg)

    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    logger.info("Preprocessed data saved to %s", output_path)

    # Save pipeline (optional)
    pipeline_path = cfg.artifacts.preprocessing_pipeline
    pipeline = build_preprocessing_pipeline(cfg)
    joblib.dump(pipeline, pipeline_path)
    logger.info("Preprocessing pipeline saved to %s", pipeline_path)


if __name__ == "__main__":
    main()