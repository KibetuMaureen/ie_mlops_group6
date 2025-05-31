"""
Feature Engineering Module

Includes:
- Age computation from DOB
- Time-based features from transaction timestamp
- Weekend indicator
- Optional: Add more domain-specific features here
"""


import pandas as pd
import logging
import sys
import yaml
from geopy.distance import geodesic

logger = logging.getLogger(__name__)


def calculate_distance(row):
    try:
        return geodesic((row["lat"], row["long"]), (row["merch_lat"], row["merch_long"])).km
    except Exception as e:
        logger.warning(f"Geo distance calculation failed for row: {e}")
        return 0.0


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate features from existing columns:
    - Age from DOB and transaction date
    - Hour, day, month from transaction datetime
    - Day of week and weekend indicator

    Parameters:
        df (pd.DataFrame): Input DataFrame (with datetime already parsed)

    Returns:
        pd.DataFrame: DataFrame with additional feature columns
    """

    df = df.copy()

    # Datetime features
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day
    df["month"] = df["trans_date_trans_time"].dt.month
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    # Age
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365
    # Geo distance
    df["geo_distance"] = df.apply(calculate_distance, axis=1)
    logger.info("Computed geo distance.")
    logger.info("Feature engineering complete.")
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    if len(sys.argv) < 2:
        logger.error(
            "Usage: python -m src.features.features <config.yaml>"
        )
        sys.exit(1)

    config_path = sys.argv[1]

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        raw_data_path = config["data_source"]["raw_path"]
        output_path = config["data_source"]["features_path"]
    except Exception as e:
        logger.exception("Failed to load config or paths: %s", e)
        sys.exit(1)

    try:
        df = pd.read_csv(raw_data_path)
        logging.info("Data loaded successfully. Shape: %s", df.shape)
    except Exception as e:
        logger.exception("Failed to load data: %s", e)
        sys.exit(1)

    try:
        df_feat = add_engineered_features(df)
        df_feat.to_csv(output_path, index=False)
        logging.info("Feature engineering complete. Saved to %s", output_path)
    except Exception as e:
        logging.exception("Feature engineering failed: %s", e)
        sys.exit(1)
