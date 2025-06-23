import sys
import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from datetime import datetime
import wandb
import pandas as pd
import tempfile
import yaml
import joblib
from dotenv import load_dotenv

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
    
from model.model import run_model_pipeline

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("model")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@hydra.main(config_path=str(PROJECT_ROOT), config_name="config", version_base=None)
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run_name = f"model_{datetime.now():%Y%m%d_%H%M%S}"

    run = wandb.init(
        project=cfg.main.WANDB_PROJECT,
        entity=cfg.main.WANDB_ENTITY,
        job_type="model_training",
        name=run_name,
        config=cfg_dict,
        tags=["model"] 
    )
    logger.info("Started WandB run: %s", run_name)

    try:
        # ─────── Load validated data from W&B artifact ───────
        artifact_name = "validated_data:latest"
        csv_filename = "validated_data.csv"

        data_art = run.use_artifact(artifact_name)
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(data_art.download(root=tmp_dir))
            df_path = data_path / csv_filename
            if not df_path.exists():
                logger.error("Expected file '%s' not found in artifact.", csv_filename)
                sys.exit(1)
            df = pd.read_csv(df_path)
            logger.info("Loaded validated data from artifact: %s", df_path)

        # Save schema and sample rows as additional artifacts
        schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
        sample_path = PROJECT_ROOT / "artifacts" / "train_sample_rows.csv"
        schema_path = PROJECT_ROOT / "artifacts" / "model_schema.json"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        df.head(50).to_csv(sample_path, index=False)
        with open(schema_path, "w") as f:
            yaml.safe_dump(schema, f)
        logger.info("Logged training schema and sample rows")

        # Log schema and sample as W&B artifact
        schema_art = wandb.Artifact("model_schema", type="schema")
        schema_art.add_file(str(schema_path))
        schema_art.add_file(str(sample_path))
        run.log_artifact(schema_art, aliases=["latest"])

        if cfg.data_load.get("log_sample_artifacts", True):
            wandb.log({"train_sample_rows": wandb.Table(dataframe=df.head(50))})

        # Run model training pipeline (returns model and preprocessor)
        model, preprocessor = run_model_pipeline(df, cfg_dict)

        # Save and log model artifact
        model_path = "model.pkl"
        joblib.dump(model, model_path)
        model_art = wandb.Artifact("model", type="model")
        model_art.add_file(model_path)
        run.log_artifact(model_art, aliases=["latest"])

        # Save and log preprocessing pipeline artifact
        preproc_path = "preprocessing_pipeline.pkl"
        joblib.dump(preprocessor, preproc_path)
        preproc_art = wandb.Artifact("preprocessing_pipeline", type="pipeline")
        preproc_art.add_file(preproc_path)
        run.log_artifact(preproc_art, aliases=["latest"])

        # Update W&B summary
        wandb.summary.update({
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "columns": list(df.columns)
        })

    except Exception as e:
        logger.exception("Model training failed")
        if run is not None:
            run.alert(title="Model Training Error", text=str(e))
        sys.exit(1)

    finally:
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")


if __name__ == "__main__":
    main()