import sys
import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from datetime import datetime
import wandb
import pandas as pd
import pickle
import json
import tempfile
from dotenv import load_dotenv

from preprocessing.preprocessing import preprocess_data, build_preprocessing_pipeline, get_output_feature_names

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("preprocessing")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def compute_df_hash(df: pd.DataFrame) -> str:
    """Compute a hash for the input DataFrame, including index."""
    import hashlib
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

@hydra.main(config_path=str(PROJECT_ROOT), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point for the preprocessing MLflow step."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"preprocess_{dt_str}"

    run = None
    try:
        run = wandb.init(
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
            job_type="preprocess",
            name=run_name,
            config=cfg_dict,
            tags=["preprocess"],
        )
        logger.info("Started WandB run: %s", run_name)

        # Load engineered data from W&B artifact
        eng_art = run.use_artifact("engineered_data:latest")
        with tempfile.TemporaryDirectory() as tmp_dir:
            eng_path = eng_art.download(root=tmp_dir)
            csv_name = Path(cfg.data_source.processed_path).name
            df = pd.read_csv(os.path.join(eng_path, csv_name))
        if df.empty:
            logger.warning("Loaded dataframe is empty.")

        # Log input data hash for traceability
        df_hash = compute_df_hash(df)
        wandb.summary["input_data_hash"] = df_hash

        # Log schema as artifact and summary
        schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
        schema_path = PROJECT_ROOT / "artifacts" / "preprocess_schema.json"
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2)
        wandb.summary["pipeline_schema"] = schema

        # Log sample input (first 50 rows)
        sample_path = PROJECT_ROOT / "artifacts" / "preprocess_sample.csv"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        df.head(50).to_csv(sample_path, index=False)
        if cfg.data_load.get("log_sample_artifacts", True):
            sample_tbl = wandb.Table(dataframe=df.head(50))
            wandb.log({"input_sample_rows": sample_tbl})

        # Log data stats (describe table)
        if cfg.data_load.get("log_summary_stats", True):
            stats_tbl = wandb.Table(dataframe=df.describe(include="all").T.reset_index())
            wandb.log({"input_stats": stats_tbl})

        # Apply label encoding to categorical columns
        df_encoded = preprocess_data(df, cfg_dict)

        # Build and fit preprocessing pipeline for numeric features
        pipeline = build_preprocessing_pipeline(cfg_dict)
        pipeline.fit(df_encoded)

        pp_path = PROJECT_ROOT / cfg.artifacts.get(
            "preprocessing_pipeline", "models/preprocessing_pipeline.pkl"
        )
        pp_path.parent.mkdir(parents=True, exist_ok=True)
        with pp_path.open("wb") as f:
            pickle.dump(pipeline, f)
        logger.info("Saved preprocessing pipeline to %s", pp_path)

        # Log pipeline artifact
        if cfg.data_load.get("log_artifacts", True):
            artifact = wandb.Artifact(
                "preprocessing_pipeline", type="pipeline"
            )
            artifact.add_file(str(pp_path))
            artifact.add_file(str(schema_path))
            artifact.add_file(str(sample_path))
            run.log_artifact(artifact, aliases=["latest"])
            logger.info("Logged preprocessing pipeline artifact to WandB")

        # ────────────────────── transform & save data ──────────────────────
        logger.info("Transforming engineered dataframe with preprocessing pipeline")
        X_proc = pipeline.transform(df_encoded)
        out_cols = get_output_feature_names(
            pipeline, df_encoded.columns.tolist(), cfg_dict
        )
        df_proc = pd.DataFrame(X_proc, columns=out_cols)
        engineered = cfg.features.get("engineered", [])
        if engineered:
            df_proc = df_proc[[c for c in engineered if c in df_proc.columns]]
        df_proc[cfg.target] = df[cfg.target]

        processed_dir = PROJECT_ROOT / cfg.artifacts.get(
            "processed_dir", "data/processed"
        )
        processed_dir.mkdir(parents=True, exist_ok=True)
        full_path = processed_dir / "preprocessed_data.csv"
        df_proc.to_csv(full_path, index=False)

        # Train/valid/test splits
        from sklearn.model_selection import train_test_split
        split_cfg = cfg.data_split
        test_size = split_cfg.get("test_size", 0.2)
        valid_size = split_cfg.get("valid_size", 0.2)
        rand_state = split_cfg.get("random_state", 42)
        train_df, temp_df = train_test_split(
            df_proc,
            test_size=(test_size + valid_size),
            random_state=rand_state,
            stratify=df_proc[cfg.target],
        )
        rel_valid = valid_size / (test_size + valid_size)
        valid_df, test_df = train_test_split(
            temp_df,
            test_size=rel_valid,
            random_state=rand_state,
            stratify=temp_df[cfg.target],
        )
        train_df.to_csv(processed_dir / "train_processed.csv", index=False)
        valid_df.to_csv(processed_dir / "valid_processed.csv", index=False)
        test_df.to_csv(processed_dir / "test_processed.csv", index=False)

        # Log processed dataset artifact
        if cfg.data_load.get("log_artifacts", True):
            data_art = wandb.Artifact("preprocessed_data", type="dataset")
            for p in [
                full_path,
                processed_dir / "train_processed.csv",
                processed_dir / "valid_processed.csv",
                processed_dir / "test_processed.csv",
            ]:
                data_art.add_file(str(p))
            run.log_artifact(data_art, aliases=["latest"])
            logger.info("Logged processed data artifact to WandB")

    except Exception as e:
        logger.exception("Failed during preprocessing step")
        if run is not None:
            run.alert(title="Preprocess Error", text=str(e))
        sys.exit(1)
    finally:
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter