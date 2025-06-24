"""
src/inferencer/run.py

This script runs batch inference using versioned artifacts from W&B.
It loads a preprocessing pipeline and a trained model, applies them
to new data, and saves the predictions.
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import joblib
from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

# --- Add project root to sys.path for absolute imports ---
# This is crucial for running MLflow steps from their subdirectories.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Now that sys.path is correct, we can import from other project modules
from features.features import add_engineered_features

# --- Basic Setup ---
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inference")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@hydra.main(
        config_path=str(PROJECT_ROOT),
        config_name="config",
        version_base=None)
def main(cfg: DictConfig):
    """
    Main function to run the inference pipeline.
    """
    run = None
    try:
        # --- Initialize W&B Run ---
        run = wandb.init(
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
            job_type="inference",
        )
        logger.info("Started WandB run for inference.")
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

        # --- Download Production Artifacts from W&B ---
        logger.info("Downloading production model and preprocessing pipeline.")

        # Download preprocessing pipeline
        preproc_art = run.use_artifact(
            f"{cfg.main.WANDB_PROJECT}/preprocessing_pipeline:latest",
            type="pipeline"
        )
        preproc_dir = Path(preproc_art.download())
        preprocessor = joblib.load(preproc_dir / "preprocessing_pipeline.pkl")
        logger.info("Loaded preprocessing pipeline artifact.")

        # Download model
        model_art = run.use_artifact(
            f"{cfg.main.WANDB_PROJECT}/model:latest", type="model"
        )
        model_dir = Path(model_art.download())
        model = joblib.load(model_dir / "model.pkl")
        logger.info("Loaded model artifact.")

        # --- Load Unseen Data ---
        # Using fraudTest.csv as the unseen data for inference
        inference_data_path = PROJECT_ROOT / "data" / "raw" / "fraudTest.csv"
        if not inference_data_path.exists():
            logger.error(
                "Inference data not found at %s", str(inference_data_path)
            )
            sys.exit(1)

        df = pd.read_csv(inference_data_path)
        logger.info("Loaded inference data with shape: %s", str(df.shape))

        # --- Preprocess Data and Generate Predictions ---
        # Select the same raw features used for training
        raw_features = cfg.get("raw_features", [])
        target = cfg.get("target")
        input_features_raw = [
            f for f in raw_features if f != target and f in df.columns
        ]

        X_raw = df[input_features_raw]
        X_raw = add_engineered_features(X_raw.copy())

        logger.info("Applying preprocessing and generating predictions...")

        X_processed = preprocessor.transform(X_raw)

        predictions = model.predict(X_processed)
        predict_proba = (
            model.predict_proba(X_processed)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        # --- Save Predictions ---
        output_df = df.copy()
        output_df["prediction"] = predictions
        if predict_proba is not None:
            output_df["prediction_proba"] = predict_proba

        output_path = PROJECT_ROOT / "data" / "inference" / "output_predictions.csv"
        output_path.parent.mkdir(exist_ok=True, parents=True)
        output_df.to_csv(output_path, index=False)

        logger.info("Saved %d predictions to %s", len(predictions), str(output_path))

        # --- Log Predictions as a W&B Artifact ---
        logger.info("Logging predictions as a W&B artifact...")
        predictions_art = wandb.Artifact(
            "inference_predictions",
            type="predictions",
            description=(
                "Predictions from the inference pipeline on unseen data."
            ),
        )
        predictions_art.add_file(str(output_path))
        run.log_artifact(predictions_art, aliases=["latest"])
        logger.info("Logged predictions artifact to W&B.")

        # Log a sample of predictions to W&B
        wandb.log({"predictions_sample": wandb.Table(dataframe=output_df.head(100))})

    except Exception as e:  # pylint: disable=broad-except
        logger.exception("Inference failed")
        if run:
            run.alert(title="Inference Error", text=str(e))
        sys.exit(1)
    finally:
        if wandb.run:
            wandb.finish()
            logger.info("WandB run finished.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
