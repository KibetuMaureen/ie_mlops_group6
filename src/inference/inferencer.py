"""
inferencer.py

Batch inference entry point.

Usage
-----
python -m src.inference.inferencer \
    data/inference/new_data.csv config.yaml data/inference/output_predictions.csv
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml

from preprocess.preprocessing import get_output_feature_names


logger = logging.getLogger(__name__)


# helper to load pickled artefacts
def _load_pickle(path: str, label: str):
    """Safely load a pickled artefact, with a descriptive error if missing"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    with p.open("rb") as fh:
        return pickle.load(fh)


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
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
    _setup_logging()

    # ── 1. Load config and artefacts ──────────────────────────────────────
    with open(config_yaml, "r", encoding="utf-8") as fh:
        config: Dict = yaml.safe_load(fh)

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

    raw_features: List[str] = config.get("raw_features", [])
    missing = [c for c in raw_features if c not in input_df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        sys.exit(1)

    X_raw = input_df[raw_features]

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


# CLI entry point
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run batch inference on a CSV file")
    parser.add_argument("input_csv", help="Path to raw input CSV")
    parser.add_argument("config_yaml", help="Path to config.yaml")
    parser.add_argument("output_csv", help="Destination for predictions CSV")
    args = parser.parse_args()

    run_inference(args.input_csv, args.config_yaml, args.output_csv)


if __name__ == "__main__":
    main()

# python -m src.inference.inferencer data/inference/new_data.csv config.yaml data/inference/output_predictions.csv
