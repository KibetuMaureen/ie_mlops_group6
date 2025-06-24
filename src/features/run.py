"""Feature engineering run script using Hydra and WandB.

Loads validated data artifact, applies feature engineering,
saves processed data, and logs results to WandB.
"""

import sys
import logging
import os
import json
from datetime import datetime
from pathlib import Path
import tempfile

import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
import pandas as pd
import wandb

# Set project root and add src to sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from features.features import add_engineered_features

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("feature_eng")


@hydra.main(
        config_path=str(PROJECT_ROOT),
        config_name="config",
        version_base=None)
def main(cfg: DictConfig) -> None:
    """Main feature engineering workflow.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"feature_eng_{dt_str}"

    run = None
    try:
        run = wandb.init(
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
            job_type="feature_engineering",
            name=run_name,
            config=cfg_dict,
            tags=["feature_engineering"],
        )
        logger.info("Started WandB run: %s", run_name)

        # Load validated data from W&B artifact
        val_art = run.use_artifact("validated_data:latest")
        with tempfile.TemporaryDirectory() as tmp_dir:
            val_path = val_art.download(root=tmp_dir)
            df = pd.read_csv(os.path.join(val_path, "validated_data.csv"))
        if df.empty:
            logger.warning("Loaded dataframe is empty.")

        # Feature engineering using add_engineered_features from features.py
        df_feat = add_engineered_features(df)

        # Save engineered data
        processed_path = PROJECT_ROOT / cfg.data_source.features_path
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df_feat.to_csv(processed_path, index=False)
        logger.info("Saved engineered data to %s", processed_path)

        # Save sample and schema
        sample_path = processed_path.parent / "engineered_sample.csv"
        df_feat.head(50).to_csv(sample_path, index=False)
        schema = {c: str(t) for c, t in df_feat.dtypes.items()}
        schema_path = processed_path.parent / "engineered_schema.json"
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)

        # Log artifacts to W&B
        if cfg.data_load.get("log_artifacts", True):
            artifact = wandb.Artifact(
                "engineered_data",
                type="dataset",
            )
            artifact.add_file(str(processed_path))
            artifact.add_file(str(sample_path))
            artifact.add_file(str(schema_path))
            run.log_artifact(artifact, aliases=["latest"])
            logger.info("Logged processed data artifact to WandB")

        if cfg.data_load.get("log_sample_artifacts", True):
            sample_tbl = wandb.Table(dataframe=df_feat.head(50))
            wandb.log({"processed_sample_rows": sample_tbl})

        wandb.summary.update(
            {
                "n_rows": df_feat.shape[0],
                "n_cols": df_feat.shape[1],
                "columns": list(df_feat.columns),
            }
        )

    except Exception as e:  # pylint: disable=broad-except
        logger.exception("Failed during feature engineering step")
        if run is not None:
            run.alert(title="Feature Eng Error", text=str(e))
        sys.exit(1)
    finally:
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
