
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import joblib

# Load the trained pipeline once on startup
# (pipeline already includes preprocessing + RandomForest)
try:
    PIPE = joblib.load("models/model.pkl")
except Exception as e:
    raise RuntimeError(f"Could not load models/model.pkl. Train first. Error: {e}")

app = FastAPI(title="Madrid House Price API", version="1.0.0")

class House(BaseModel):
    m2: Optional[float] = None
    rooms: Optional[float] = None
    elevator: Optional[float] = Field(default=None, description="0/1")
    garage: Optional[float] = Field(default=None, description="0/1")
    house_type: Optional[str] = None
    house_type_2: Optional[str] = None
    neighborhood: Optional[str] = None
    district: Optional[str] = None

    # Accept bools/ints and coerce to float 0/1 for model compatibility
    @validator("elevator", "garage", pre=True)
    def _coerce_bool01(cls, v):
        if v is None:
            return v
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        try:
            return float(v)
        except Exception:
            return None

@app.get("/healthz")
def health():
    return {"status": "ok", "model_loaded": True}

@app.get("/schema")
def schema():
    return {
        "numeric": ["m2", "rooms", "elevator", "garage"],
        "categorical": ["house_type", "house_type_2", "neighborhood", "district"],
        "notes": "All fields optional; pipeline imputes/ignores missing values."
    }

@app.post("/predict")
def predict(items: List[House]):
    if not items:
        raise HTTPException(status_code=400, detail="Empty payload")
    df = pd.DataFrame([i.dict() for i in items])

    try:
        log_preds = PIPE.predict(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    prices = np.expm1(log_preds)  # inverse of log1p
    return {
        "count": len(items),
        "log_price": [float(x) for x in log_preds],
        "price": [float(x) for x in prices],
    }

# http://127.0.0.1:8000/docs#/default/predict_predict_post
[
  {
    "m2": 85,
    "rooms": 3,
    "elevator": 1,
    "garage": 0,
    "house_type": "piso",
    "house_type_2": "reformado",
    "neighborhood": "sol",
    "district": "centro"
  }
]
