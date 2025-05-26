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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
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
    try:
        logger.info("Starting feature engineering...")

        # Compute age in years
        df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365
        logger.info("Computed age from DOB")

        # Extract time-based features
        df["hour"] = df["trans_date_trans_time"].dt.hour
        df["day"] = df["trans_date_trans_time"].dt.day
        df["month"] = df["trans_date_trans_time"].dt.month
        df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        logger.info("Extracted hour, day, month, day_of_week, and is_weekend")

        logger.info("Feature engineering completed successfully.")
        return df

    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise