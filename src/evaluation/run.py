"""
evaluation/run.py

MLflow-compatible evaluation step using versioned artifacts from W&B.
"""

import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import wandb
import pandas as pd
import joblib
from dotenv import load_dotenv

# --- Add project root to sys.path for absolute imports ---
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from features.features import add_engineered_features
from evaluation.evaluator_sklearn import evaluate_model

# --- Basic Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluation")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@hydra.main(config_path=str(PROJECT_ROOT), config_name="config", version_base=None)
def main(cfg: DictConfig):
    run = None
    try:
        run = wandb.init(
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
            job_type="evaluation"
        )
        logger.info("Started WandB run for evaluation.")
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

        # --- Download All Necessary Artifacts from W&B ---
        logger.info("Downloading production artifacts...")
        
        # Model
        model_art = run.use_artifact(f"{cfg.main.WANDB_PROJECT}/model:latest", type="model")
        model_dir = model_art.download()
        model = joblib.load(Path(model_dir) / "model.pkl")
        logger.info("Loaded model artifact.")

        # Preprocessing Pipeline
        preproc_art = run.use_artifact(f"{cfg.main.WANDB_PROJECT}/preprocessing_pipeline:latest", type="pipeline")
        preproc_dir = Path(preproc_art.download())
        preprocessor = joblib.load(preproc_dir / "preprocessing_pipeline.pkl")
        logger.info("Loaded preprocessing pipeline artifact.")

        # Raw Test Data
        splits_art = run.use_artifact(f"{cfg.main.WANDB_PROJECT}/data_splits:latest", type="dataset")
        splits_dir = Path(splits_art.download())
        X_test_raw = pd.read_csv(splits_dir / "X_test.csv")
        y_test = pd.read_csv(splits_dir / "y_test.csv")["target"]
        logger.info("Loaded raw test data from artifact.")

        # --- Replicate the Training Data Pipeline ---
        logger.info("Applying feature engineering and preprocessing to test data...")
        X_test_eng = add_engineered_features(X_test_raw.copy())
        
        X_test_pp = pd.DataFrame(
            preprocessor.transform(X_test_eng),
            columns=preprocessor.get_feature_names_out()
        )

        # Select the same final features used for training
        final_features = cfg.get("final_features", [])
        if not final_features:
            raise ValueError("`final_features` not defined in config.")
        
        X_test_final = X_test_pp[final_features]
        logger.info(f"Prepared final test data with shape: {X_test_final.shape}")

        # --- Generate Predictions and Evaluate ---
        y_pred = model.predict(X_test_final)
        metrics = evaluate_model(y_test, y_pred)
        logger.info(f"Evaluation metrics: {metrics}")

        wandb.log(metrics)
        wandb.summary.update(metrics)

    except Exception as e:
        logger.exception("Evaluation failed")
        if run:
            run.alert(title="Evaluation Error", text=str(e))
        sys.exit(1)
    finally:
        if wandb.run:
            wandb.finish()
            logger.info("WandB run finished.")

if __name__ == "__main__":
    main()