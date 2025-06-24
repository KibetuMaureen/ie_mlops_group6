"""
data_loader/run.py

MLflow-compatible, modular data loading step with Hydra config,
W&B artifact logging, and robust error handling.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
import wandb

from data_loader import get_data

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("data_load")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@hydra.main(config_path=str(PROJECT_ROOT), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for data loading and logging with W&B.
    """
    config_path = PROJECT_ROOT / "config.yaml"
    output_dir = PROJECT_ROOT / cfg.data_load.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path_cfg = Path(cfg.data_source.raw_path)
    resolved_raw_path = (
        raw_path_cfg if raw_path_cfg.is_absolute()
        else PROJECT_ROOT / raw_path_cfg
    )

    print("Looking for file at:", resolved_raw_path)
    if not resolved_raw_path.is_file():
        logger.error(
            "Data file does not exist: %s", resolved_raw_path
        )
        raise FileNotFoundError(
            f"Data file not found: {resolved_raw_path}"
        )

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = resolved_raw_path.name
    run_name = f"data_load_{dt_str}_{data_file}"

    run = None
    try:
        run = wandb.init(
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
            job_type="data_load",
            name=run_name,
            config=dict(cfg),
            tags=["data_load", data_file]
        )
        logger.info("Started WandB run: %s", run_name)

        data_stage = cfg.data_load.data_stage
        df = get_data(
            config_path=str(config_path),
            data_stage=data_stage,
            env_path=None
        )
        if df.empty:
            logger.warning(
                "Loaded dataframe is empty: %s", resolved_raw_path
            )
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            logger.warning(
                "Duplicates found in data (%d rows). "
                "Consider removing them before use.", dup_count
            )

        # W&B logging (conditional via config)
        if cfg.data_load.get("log_sample_artifacts", True):
            sample_tbl = wandb.Table(dataframe=df.head(100))
            wandb.log({"sample_rows": sample_tbl})

        if cfg.data_load.get("log_summary_stats", True):
            stats_tbl = wandb.Table(
                dataframe=df.describe(include="all").T.reset_index()
            )
            wandb.log({"summary_stats": stats_tbl})

        if cfg.data_load.get("log_artifacts", True):
            raw_art = wandb.Artifact("raw_data", type="dataset")
            raw_art.add_file(str(resolved_raw_path), name="raw_data.csv")
            run.log_artifact(raw_art, aliases=["latest"])
            logger.info("Logged raw data artifact to WandB")

        wandb.summary.update({
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "n_duplicates": dup_count,
            "columns": list(df.columns)
        })

    except Exception as e:  # pylint: disable=broad-except
        logger.exception("Failed during data loading step")
        if run is not None:
            run.alert(title="Data Load Error", text=str(e))
        sys.exit(1)
    finally:
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
