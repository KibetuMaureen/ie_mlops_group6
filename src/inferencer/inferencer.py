"""
inferencer.py

Batch inference entry point.

"""

from __future__ import annotations

import os
import argparse
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.preprocessing.preprocessing import (
    get_output_feature_names,
)

from src.features.features import add_engineered_features


logger = logging.getLogger(__name__)


# helper to load pickled artefacts
def _load_pickle(path: str, label: str):
    """Safely load a pickled artefact, with a descriptive error if missing"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    with p.open("rb") as fh:
        return pickle.load(fh)


def setup_logging(logging_config: dict):
    """
    Set up logging to both console and file using configuration from YAML.

    Args:
        logging_config (dict):

        Configuration dictionary with keys like 'log_file',
        'level', 'format', and 'datefmt'.
    """
    log_file = logging_config.get("log_file", "logs/main.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    log_format = logging_config.get(
        "format", "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    date_format = logging_config.get("datefmt", "%Y-%m-%d %H:%M:%S")
    log_level = getattr(
        logging,
        logging_config.get("level", "INFO").upper(),
        logging.INFO)

    # Remove all existing handlers to prevent duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create and configure file handler
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    # Create and configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))

    # Apply the handlers
    logging.basicConfig(
        level=log_level,
        handlers=[file_handler, console_handler]
        )


def run_inference(input_csv: str, config_yaml: str, output_csv: str) -> None:
    """
    Run batch inference:
    1. Load config, preprocessing pipeline, and trained model
    2. Validate that required **raw_features** exist in the input CSV
    3. Transform features via the pipeline
    4. Optionally keep only the engineered subset used during training
    5. Generate predictions and save to CSV
    """

    # ── 1. Load config and artefacts ──────────────────────────────────────
    with open(config_yaml, "r", encoding="utf-8") as fh:
        config: dict = yaml.safe_load(fh)

    # Setup logging using config
    setup_logging(config.get("logging", {}))

    pp_path = config.get("artifacts", {}).get(
        "preprocessing_pipeline", "models/preprocessing_pipeline.pkl"
    )
    model_path = config.get("artifacts", {}).get(
        "model_path", "models/model.pkl")

    logger.info("Loading preprocessing pipeline: %s", pp_path)
    pipeline = _load_pickle(pp_path, "preprocessing pipeline")

    logger.info("Loading trained model: %s", model_path)
    model = _load_pickle(model_path, "model")

    # ── 2. Read raw data and basic validation ─────────────────────────────
    logger.info("Reading input CSV: %s", input_csv)
    input_df: pd.DataFrame = pd.read_csv(input_csv)
    logger.info("Input shape: %s", input_df.shape)

    # raw_features: List[str] = config.get("raw_features", [])
    raw_features = config.get("raw_features", [])
    target = config["target"]
    input_features_raw = [f for f in raw_features if f != target]

    missing = [c for c in raw_features if c not in input_df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        sys.exit(1)

    X_raw = input_df[input_features_raw]
    X_raw = add_engineered_features(X_raw.copy())

    # ── 3. Transform via the *same* preprocessing pipeline ────────────────
    logger.info("Applying preprocessing pipeline to input data")
    X_proc = pipeline.transform(X_raw)

    # ── 4. Keep only engineered features that were used in training ───────
    engineered = config.get("features", {}).get("engineered", [])
    if engineered:
        feature_names = get_output_feature_names(
            preprocessor=pipeline,
            input_features=raw_features,
            config=config,
        )
        selected = [f for f in engineered if f in feature_names]
        if not selected:
            logger.error(
                "None of the engineered features are present after transform"
            )
            sys.exit(1)
        indices = [feature_names.index(f) for f in selected]
        X_proc = X_proc[:, indices]

    # ── 5. Generate predictions ───────────────────────────────────────────
    logger.info("Generating predictions")
    input_df["prediction"] = model.predict(X_proc)
    if hasattr(model, "predict_proba"):
        input_df["prediction_proba"] = model.predict_proba(X_proc)[:, 1]

    # ── 6. Save results ───────────────────────────────────────────────────
    logger.info("Writing predictions to %s", output_csv)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    input_df.to_csv(output_csv, index=False)
    logger.info("Inference complete")


def run_inference_df(df: pd.DataFrame, config: dict, pipeline=None, model=None) -> pd.DataFrame:
    """
    In-memory batch inference:
    1. Validate required raw_features exist in df
    2. Apply feature engineering
    3. Transform features via the pipeline
    4. Optionally keep only engineered features
    5. Generate predictions and probabilities
    Returns a DataFrame with predictions and probabilities.
    """
    # Load pipeline and model if not provided
    if pipeline is None:
        pp_path = config.get("artifacts", {}).get(
            "preprocessing_pipeline", "models/preprocessing_pipeline.pkl"
        )
        pipeline = _load_pickle(pp_path, "preprocessing pipeline")
    if model is None:
        model_path = config.get("artifacts", {}).get(
            "model_path", "models/model.pkl")
        model = _load_pickle(model_path, "model")

    raw_features = config.get("raw_features", [])
    target = config.get("target", None)
    input_features_raw = [f for f in raw_features if f != target]

    missing = [c for c in raw_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X_raw = df[input_features_raw]
    X_raw = add_engineered_features(X_raw.copy())

    X_proc = pipeline.transform(X_raw)

    engineered = config.get("features", {}).get("engineered", [])
    if engineered:
        feature_names = get_output_feature_names(
            preprocessor=pipeline,
            input_features=raw_features,
            config=config,
        )
        selected = [f for f in engineered if f in feature_names]
        if not selected:
            raise ValueError("None of the engineered features are present after transform")
        indices = [feature_names.index(f) for f in selected]
        X_proc = X_proc[:, indices]

    preds = model.predict(X_proc)
    proba = model.predict_proba(X_proc)[:, 1] if hasattr(model, "predict_proba") else None

    result_df = df.copy()
    result_df["prediction"] = preds
    if proba is not None:
        result_df["prediction_proba"] = proba
    return result_df


# CLI entry point
def main() -> None:
    """
    Parse command-line arguments and run batch inference on data.

    Expects three positional arguments:
    1. Path to raw input CSV file with data for inference.
    2. Path to config YAML file specifying model parameters.
    3. Path to output CSV file for saving predictions.

    Parses arguments and calls run_inference with them.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Run batch inference on a CSV file")
    parser.add_argument("input_csv", help="Path to raw input CSV")
    parser.add_argument("config_yaml", help="Path to config.yaml")
    parser.add_argument("output_csv", help="Destination for predictions CSV")
    args = parser.parse_args()

    run_inference(args.input_csv, args.config_yaml, args.output_csv)


if __name__ == "__main__":
    main()
