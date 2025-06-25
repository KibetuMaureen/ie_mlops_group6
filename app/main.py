from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import pickle
import os
import yaml
from pathlib import Path

from src.features.features import add_engineered_features
from src.preprocessing.preprocessing import get_output_feature_names
from src.inferencer.inferencer import run_inference_df

# Project root and config
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as fh:
    CONFIG = yaml.safe_load(fh)

PIPELINE_PATH = PROJECT_ROOT / CONFIG.get("artifacts", {}).get(
    "preprocessing_pipeline", "models/preprocessing_pipeline.pkl"
)
MODEL_PATH = PROJECT_ROOT / CONFIG.get("artifacts", {}).get(
    "model_path", "models/model.pkl"
)

with PIPELINE_PATH.open("rb") as fh:
    PIPELINE = pickle.load(fh)
with MODEL_PATH.open("rb") as fh:
    MODEL = pickle.load(fh)

RAW_FEATURES = CONFIG.get("raw_features", [])
ENGINEERED = CONFIG.get("features", {}).get("engineered", [])

app = FastAPI()

# Build input schema from raw_features
class TransactionInput(BaseModel):
    trans_date_trans_time: str = Field(..., example="2020-01-01 00:00:00")
    category: str = Field(..., example="shopping")
    amt: float = Field(..., example=100.0)
    gender: str = Field(..., example="M")
    city: str = Field(..., example="New York")
    state: str = Field(..., example="NY")
    zip: int = Field(..., example=10001)
    lat: float = Field(..., example=40.7128)
    long: float = Field(..., example=-74.0060)
    city_pop: int = Field(..., example=8000000)
    job: str = Field(..., example="Engineer")
    dob: str = Field(..., example="1980-01-01")
    merch_lat: float = Field(..., example=40.7128)
    merch_long: float = Field(..., example=-74.0060)

    class Config:
        schema_extra = {
            "example": {
                "trans_date_trans_time": "2020-01-01 00:00:00",
                "category": "shopping",
                "amt": 100.0,
                "gender": "M",
                "city": "New York",
                "state": "NY",
                "zip": 10001,
                "lat": 40.7128,
                "long": -74.0060,
                "city_pop": 8000000,
                "job": "Engineer",
                "dob": "1980-01-01",
                "merch_lat": 40.7128,
                "merch_long": -74.0060
            }
        }

@app.get("/")
def root():
    return {"message": "Welcome to the credit card fraud prediction API"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: TransactionInput):
    try:
        df = pd.DataFrame([data.dict()])
        result_df = run_inference_df(df, CONFIG, pipeline=PIPELINE, model=MODEL)
        pred = int(result_df["prediction"].iloc[0])
        proba = float(result_df["prediction_proba"].iloc[0]) if "prediction_proba" in result_df else None
        return {"prediction": pred, "probability": proba}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
def predict_batch(data: List[TransactionInput]):
    try:
        df = pd.DataFrame([d.dict() for d in data])
        result_df = run_inference_df(df, CONFIG, pipeline=PIPELINE, model=MODEL)
        result = result_df[["prediction"] + (["prediction_proba"] if "prediction_proba" in result_df else [])]
        result = result.rename(columns={"prediction_proba": "probability"})
        return result.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 