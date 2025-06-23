from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model

app = FastAPI(title="Crime Risk Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia a ["http://localhost:3000"] si quieres restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Input Models ===

class PredictionInput(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    date: str

    @validator("date")
    def validate_date(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format.")

class BatchPredictionInput(BaseModel):
    inputs: List[PredictionInput]

# === Output Model ===

class PredictionOutput(BaseModel):
    risk_score: float
    risk_level: str
    probability_distribution: dict
    confidence: float

# === Carga de modelos y artefactos ===

MODEL_DIR = "exported_models"
try:
    fusion_model = joblib.load(os.path.join(MODEL_DIR, "fusion_model.pkl"))
    scaler, expected_columns = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))  # scaler + columnas
    rnn_model = load_model(os.path.join(MODEL_DIR, "rnn_model.keras"))  # O .h5 si corresponde
    model_status = "ready"
except Exception as e:
    fusion_model = None
    scaler = None
    expected_columns = None
    rnn_model = None
    model_status = f"error: {e}"

# === Funciones auxiliares ===

def preprocess_input(lat, lon, date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")

    # Rellenamos las columnas esperadas con 0
    data = {col: 0 for col in expected_columns}

    # Asignamos los valores que tenemos si están en expected_columns
    if "latitude" in data: data["latitude"] = lat
    if "longitude" in data: data["longitude"] = lon
    if "month" in data: data["month"] = dt.month
    if "day" in data: data["day"] = dt.day
    if "hour" in data: data["hour"] = 12  # puedes cambiar si tienes el dato real
    if "day_of_week" in data: data["day_of_week"] = dt.weekday()

    df = pd.DataFrame([data])
    return df[expected_columns]

def categorize_risk(score):
    if score < 0.33:
        return "LOW"
    elif score < 0.66:
        return "MEDIUM"
    else:
        return "HIGH"

# === Endpoints ===

@app.get("/model/status")
def get_model_status():
    return {"model_status": model_status}

@app.post("/predict", response_model=PredictionOutput)
def predict_risk(input_data: PredictionInput):
    if model_status != "ready":
        raise HTTPException(status_code=503, detail="Model not ready")

    df_input = preprocess_input(input_data.latitude, input_data.longitude, input_data.date)
    X_scaled = scaler.transform(df_input)
    dummy_embedding = np.zeros((1, 15))  # Simula vector RNN si no usas secuencias reales

    input_combined = np.hstack([X_scaled, dummy_embedding])
    proba = fusion_model.predict_proba(input_combined)[0]
    risk_score = proba.max()
    risk_level = categorize_risk(risk_score)
    confidence = risk_score

    return PredictionOutput(
        risk_score=round(risk_score, 4),
        risk_level=risk_level,
        probability_distribution={str(i): round(p, 4) for i, p in enumerate(proba)},
        confidence=round(confidence, 4)
    )

@app.post("/predict/batch", response_model=List[PredictionOutput])
def predict_batch(batch: BatchPredictionInput):
    results = []
    for entry in batch.inputs:
        try:
            res = predict_risk(entry)
            results.append(res)
        except Exception:
            results.append(PredictionOutput(
                risk_score=0.0,
                risk_level="UNKNOWN",
                probability_distribution={},
                confidence=0.0
            ))
    return results

@app.post("/data/update")
def update_data():
    return {"status": "update endpoint not implemented yet"}
