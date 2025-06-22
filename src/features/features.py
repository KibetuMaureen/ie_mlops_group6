"""
Feature Engineering Module with Hydra

Includes:
- Age computation from DOB
- Time-based features from transaction timestamp
- Weekend indicator
- Geo distance
"""

import pandas as pd
import logging
from geopy.distance import geodesic
import hydra
from omegaconf import DictConfig
import os

logger = logging.getLogger(__name__)


def calculate_distance(row):
    try:
        return geodesic((row["lat"], row["long"]), (row["merch_lat"], row["merch_long"])).km
    except Exception as e:
        logger.warning(f"Geo distance calculation failed for row: {e}")
        return 0.0


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day
    df["month"] = df["trans_date_trans_time"].dt.month
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365

    df["geo_distance"] = df.apply(calculate_distance, axis=1)
    logger.info("Feature engineering complete.")
    return df


@hydra.main(config_path='../../conf', config_name='config', version_base='1.3')
def main(cfg: DictConfig):
    logging.basicConfig(
        level=cfg.logging.level,
        format=cfg.logging.format,
        datefmt=cfg.logging.datefmt,
        filename=cfg.logging.log_file
    )

    raw_data_path = cfg.data_source.processed_path
    output_path = cfg.data_source.features_path

    try:
        df = pd.read_csv(raw_data_path)
        logger.info("Data loaded successfully. Shape: %s", df.shape)
    except Exception as e:
        logger.exception("Failed to load data: %s", e)
        raise

    try:
        df_feat = add_engineered_features(df)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_feat.to_csv(output_path, index=False)
        logger.info("Feature engineering complete. Saved to %s", output_path)
    except Exception as e:
        logger.exception("Feature engineering failed: %s", e)
        raise


if __name__ == "__main__":
    main()