import mlflow
import tempfile
import os
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from datetime import datetime
import wandb
import yaml
import shutil
from pathlib import Path
import logging

load_dotenv()

# Basic logger for pre-run tasks
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pre-run cache cleanup ---
# MLflow and WandB can aggressively cache artifacts. To ensure we use the
# latest versions from a new run, we can clear the cache directories.
WANDB_CACHE_DIR = Path.home() / ".cache" / "wandb"
if WANDB_CACHE_DIR.exists():
    logger.info("Clearing W&B artifact cache at: %s", WANDB_CACHE_DIR)
    shutil.rmtree(WANDB_CACHE_DIR, ignore_errors=True)

# List of pipeline steps in order
PIPELINE_STEPS = [
    "data_loader",
    "data_validation",
    "model",
    "inferencer",
]

# Steps that accept Hydra overrides via MLflow parameters (if any)
STEPS_WITH_OVERRIDES = {"model"}

@hydra.main(config_name="config", config_path=".", version_base=None)
def main(cfg: DictConfig):
    os.environ["WANDB_PROJECT"] = cfg.main.WANDB_PROJECT
    os.environ["WANDB_ENTITY"] = cfg.main.WANDB_ENTITY

    run_name = f"orchestrator_{datetime.now():%Y%m%d_%H%M%S}"
    run = wandb.init(
        project=cfg.main.WANDB_PROJECT,
        entity=cfg.main.WANDB_ENTITY,
        job_type="orchestrator",
        name=run_name,
    )
    print(f"Started WandB run: {run.name}")

    steps_raw = cfg.main.steps
    active_steps = [s.strip() for s in steps_raw.split(",") if s.strip()] \
        if steps_raw != "all" else PIPELINE_STEPS
    
    #hydra_override = getattr(cfg.main, "hydra_options", "").strip()
    hydra_override = cfg.main.hydra_options if hasattr(
        cfg.main, "hydra_options") else ""
    # Only pass hydra_options if it's not empty and the step supports it

    with tempfile.TemporaryDirectory():
        for step in active_steps:
            step_dir = os.path.join(
                hydra.utils.get_original_cwd(), "src", step)

            params = {}
            if step in STEPS_WITH_OVERRIDES and hydra_override:
                params["hydra_options"] = hydra_override

            print(f"Running step: {step} (W&B run: {run.name})")
            if params:
                mlflow.run(step_dir, "main", parameters=params)
            else:
                mlflow.run(step_dir, "main")

    wandb.finish()

if __name__ == "__main__":
    main()

# force render deploy
