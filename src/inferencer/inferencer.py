""""
inferencer.py

Loads config.yaml for paths & settings, then runs batch inference:
  - Validates input against schema
  - Applies full preprocessing pipeline to the input data
  - Predicts class & probability using the active model
  - Logs every inference call for auditability

CLI:
  python -m src.inference.inferencer \
    data/inference/newdata_fraudtest.csv \
    configs/config.yaml \
    data/inference/output_predictions.csv
"""

import os
import sys
import logging
import pickle
import yaml
import argparse
from datetime import datetime
from typing import List, Dict

import pandas as pd
#from pydantic import BaseModel, ValidationError
#from preprocess.preprocessing import get_output_feature_names #check path 

# ------------------------------------------------------------------------------
# 1) CONFIGURATION & LOGGING
# ------------------------------------------------------------------------------

CONFIG_PATH = "../configs/config.yaml"  

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# Inspect a few keys:
print("Active model:", cfg["model"]["active"])
print("Model save path:", cfg["model"][cfg["model"]["active"]]["save_path"])
print("Pipeline path:", cfg["artifacts"]["preprocessing_pipeline"])
print("Inference threshold (default 0.5):", cfg.get("inference", {}).get("threshold", 0.5)) #review if we want to change it

# Ensure the config has the expected structure
if "model" not in cfg or "active" not in cfg["model"]:
    raise ValueError("Config must specify an active model under 'model.active'")

# Active model key 
active = cfg["model"]["active"]

# Paths for model and pipeline artifacts
MODEL_FILE    = cfg["model"][active]["save_path"]
PIPELINE_FILE = cfg["artifacts"]["preprocessing_pipeline"]

# Inference threshold (can be added under a new 'inference' section in config) review if we want to change it 
THRESHOLD = cfg.get("inference", {}).get("threshold", 0.5) 

# set up audit logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler("logs/inference.log")
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(fh)

# Function to load pickled artifacts with error handling
def _load_pickle(path: str, label: str):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{label} not found at {path}")
    with p.open("rb") as fh:
        return pickle.load(fh)

def _setup_logging():
    log_cfg = cfg["logging"]
    logging.basicConfig(
        filename=log_cfg["log_file"],
        level=log_cfg["level"],
        format=log_cfg["format"],
        datefmt=log_cfg.get("datefmt"),
    )
    logger = logging.getLogger(__name__)


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


    def run_inference(input_csv: str, config_yaml: str, output_csv: str) -> None:
    # 1) Load config
    with open(config_yaml, "r", encoding="utf-8") as fh:
        cfg: Dict = yaml.safe_load(fh)

    active = cfg["model"]["active"]                                        # "xgboost"
    model_path = cfg["model"][active]["save_path"]                         # :contentReference[oaicite:4]{index=4}
    pp_path    = cfg["artifacts"]["preprocessing_pipeline"]                # :contentReference[oaicite:5]{index=5}
    threshold  = cfg.get("inference", {}).get("threshold", 0.5)

    logger.info("Active model: %s", active)
    logger.info("Loading pipeline from %s", pp_path)
    pipeline = _load_pickle(pp_path, "preprocessing pipeline")

    logger.info("Loading model from %s", model_path)
    model = _load_pickle(model_path, "model")


    # 2) Read raw data and validate ─────────────────────────────
    logger.info("Reading input CSV: %s", input_csv)
    input_df: pd.DataFrame = pd.read_csv(input_csv)
    logger.info("Input shape: %s", input_df.shape)

    raw_features: List[str] = config.get("raw_features", [])
    missing = [c for c in raw_features if c not in input_df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        sys.exit(1)

    X_raw = input_df[raw_features]

    # 3) Transform via the preprocessing pipeline ────────────────
    logger.info("Applying preprocessing pipeline")
    try:
        X_proc = pipeline.transform(X_raw)
    except Exception:
        logger.exception("Preprocessing failed")
        sys.exit(1)

    # 4) Keep only engineered features that were used in training ───────
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

    # 5) Generate predictions ───────────────────────────────────────────
    logger.info("Generating predictions")
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_proc)[:, 1]
        preds = (probs >= threshold).astype(int)
    else:
        preds = model.predict(X_proc)
        probs = None

    df["prediction"] = preds
    if probs is not None:
        df["prediction_proba"] = probs

    # 6) Save results ───────────────────────────────────────────────────
    logger.info("Writing predictions to %s", output_csv)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    input_df.to_csv(output_csv, index=False)
    logger.info("Inference complete")



# CLI entry point
def main() -> None:
    parser = argparse.ArgumentParser(description="Batch inference on CSV")
    parser.add_argument("input_csv",   help="Raw input CSV path")
    parser.add_argument("config_yaml", help="Config YAML path")
    parser.add_argument("output_csv",  help="Output predictions CSV path")
    args = parser.parse_args()
    run_inference(args.input_csv, args.config_yaml, args.output_csv)

if __name__ == "__main__":
    main()
  