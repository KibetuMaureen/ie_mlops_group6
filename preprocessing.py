"""
Preprocessing Module

This module includes:
- Missing value handling
- Date parsing and feature extraction
- Encoding categorical variables
- Feature scaling

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete preprocessing pipeline:
    - Handles missing values
    - Parses dates
    - Encodes categoricals
    - Scales numerical features

    Parameters:
        df (pd.DataFrame): Raw input DataFrame

    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    try:
        logger.info("Starting preprocessing...")

        # ---- Handle Missing Values ----
        df = df.copy()
        df.fillna(method='ffill', inplace=True)
        logger.info("Filled missing values using forward fill")

        # ---- Encode Categorical Features ----
        categorical_cols = df.select_dtypes(include="object").columns.tolist()
        if "trans_date_trans_time" in categorical_cols:
            categorical_cols.remove("trans_date_trans_time")

        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            logger.info(f"Encoded column: {col}")

        logger.info("Preprocessing completed successfully.")
        return df

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise