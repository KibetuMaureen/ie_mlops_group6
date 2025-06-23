import os
import logging
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig
import hydra

# Set up top-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_data(path: str, delimiter: str, header: int, encoding: str) -> pd.DataFrame:
    if not path or not isinstance(path, str):
        logger.error("Invalid path.")
        raise ValueError("Invalid path.")
    if not os.path.isfile(path):
        logger.error("File not found: %s", path)
        raise FileNotFoundError(f"File not found: {path}")
    try:
        df = pd.read_csv(path, delimiter=delimiter, header=header, encoding=encoding)
        logger.info("Loaded data from %s, shape=%s", path, df.shape)
        return df
    except Exception as e:
        logger.exception("Failed to load data: %s", e)
        raise

@hydra.main(config_path="../../", config_name="config", version_base=None)

def main(cfg: DictConfig):
    load_dotenv(override=True)  # Load environment variables

    # Determine data path based on what's needed
    path = cfg.data_source.raw_path
    delimiter = cfg.data_source.delimiter
    header = cfg.data_source.header
    encoding = cfg.data_source.encoding

    # Load the data
    df = load_data(path, delimiter, header, encoding)

    # Optional: validate required schema
    if cfg.get("data_validation", {}).get("enabled", False):
        expected_cols = [col["name"] for col in cfg.data_validation.schema.columns]
        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            if cfg.data_validation.action_on_error == "raise":
                raise ValueError(f"Missing required columns: {missing}")
            else:
                logger.warning("Missing required columns: %s", missing)

    logger.info("Data loaded and validated.")
    print(df.head())

if __name__ == "__main__":
    main()


