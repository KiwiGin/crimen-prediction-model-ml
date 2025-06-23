from fastapi import FastAPI, HTTPException
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import os
import time
import sqlite3
import json
from shapely.geometry import Point, shape
from tensorflow.keras.models import load_model, Model
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI(title="Crime Risk Prediction API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=10)

# === Modelos de entrada y salida ===
class ClassRiskMapRequest(BaseModel):
    crime_class_index: int
    date: str

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

class PredictionOutput(BaseModel):
    risk_score: float
    risk_level: str
    probability_distribution: dict
    confidence: float
    latitude: float
    longitude: float
    date: str
    model_version: str
    processing_time_ms: int
    temporal_score: float
    spatial_score: float

# === Carga de modelos ===
MODEL_DIR = "exported_models"
try:
    fusion_model = joblib.load(os.path.join(MODEL_DIR, "fusion_model.pkl"))
    scaler, expected_columns = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    full_rnn_model = load_model(os.path.join(MODEL_DIR, "rnn_model.keras"))
    rnn_model = Model(
        inputs=full_rnn_model.input,
        outputs=full_rnn_model.get_layer("dense_embedding").output
    )
    model_status = "ready"
except Exception as e:
    fusion_model = None
    scaler = None
    expected_columns = None
    rnn_model = None
    model_status = f"error: {e}"

# === Preprocesamiento ===
def preprocess_input(lat, lon, date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    data = {col: 0 for col in expected_columns}
    if "latitude" in data: data["latitude"] = lat
    if "longitude" in data: data["longitude"] = lon
    if "month" in data: data["month"] = dt.month
    if "day" in data: data["day"] = dt.day
    if "hour" in data: data["hour"] = 12
    if "day_of_week" in data: data["day_of_week"] = dt.weekday()
    df = pd.DataFrame([data])
    return df[expected_columns]

def get_rnn_embedding(lat: float, lon: float, date_str: str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    sequence = []
    for i in range(30):
        day = dt - pd.Timedelta(days=29 - i)
        sequence.append([lat, lon, day.weekday()])
    sequence = np.array(sequence).reshape(1, 30, 3)
    return rnn_model.predict(sequence, verbose=0)

def categorize_risk(score):
    if score < 0.33:
        return "LOW"
    elif score < 0.66:
        return "MEDIUM"
    else:
        return "HIGH"

def save_prediction_to_db(data: PredictionOutput):
    try:
        conn = sqlite3.connect("predictions.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                latitude REAL,
                longitude REAL,
                date TEXT,
                risk_score REAL,
                risk_level TEXT,
                probability_distribution TEXT,
                confidence REAL,
                model_version TEXT,
                processing_time_ms INTEGER,
                temporal_score REAL,
                spatial_score REAL
            )
        """)
        cursor.execute("""
            INSERT INTO prediction_history (
                latitude, longitude, date,
                risk_score, risk_level,
                probability_distribution, confidence,
                model_version, processing_time_ms,
                temporal_score, spatial_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.latitude,
            data.longitude,
            data.date,
            data.risk_score,
            data.risk_level,
            json.dumps(data.probability_distribution),
            data.confidence,
            data.model_version,
            data.processing_time_ms,
            data.temporal_score,
            data.spatial_score
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[WARN] Error al guardar en la base de datos: {e}")

# === Predicción ===
def generate_prediction(input_data: PredictionInput, save: bool = True) -> PredictionOutput:
    df_input = preprocess_input(input_data.latitude, input_data.longitude, input_data.date)
    X_scaled = scaler.transform(df_input)

    start_time = time.time()
    rnn_embedding = get_rnn_embedding(input_data.latitude, input_data.longitude, input_data.date)
    input_combined = np.hstack([X_scaled, rnn_embedding.reshape(1, -1)])

    proba = fusion_model.predict_proba(input_combined)[0]
    elapsed_ms = int((time.time() - start_time) * 1000)

    risk_score = float(proba.max())
    risk_level = categorize_risk(risk_score)
    confidence = risk_score

    temporal_score = float(X_scaled[0][expected_columns.index("day")]) if "day" in expected_columns else 0
    spatial_score = float(X_scaled[0][expected_columns.index("latitude")]) if "latitude" in expected_columns else 0

    output = PredictionOutput(
        risk_score=round(risk_score, 4),
        risk_level=risk_level,
        probability_distribution={str(i): round(p, 4) for i, p in enumerate(proba)},
        confidence=round(confidence, 4),
        latitude=input_data.latitude,
        longitude=input_data.longitude,
        date=input_data.date,
        model_version="1.1",
        processing_time_ms=elapsed_ms,
        temporal_score=temporal_score,
        spatial_score=spatial_score,
    )

    if save:
        save_prediction_to_db(output)

    return output

def generate_grid(min_lat, max_lat, min_lon, max_lon, step=0.01):
    latitudes = np.arange(min_lat, max_lat, step)
    longitudes = np.arange(min_lon, max_lon, step)
    return [(lat, lon) for lat in latitudes for lon in longitudes]
# Cargar zona urbana (puede ser un distrito, ciudad, etc.)
with open("atlanta.geojson") as f:
    urban_polygon = shape(json.load(f)["features"][0]["geometry"])

def is_in_urban_area(lat, lon):
    return urban_polygon.contains(Point(lon, lat))

# === Endpoint con threads ===
@app.post("/predict/class-risk-map")
def class_risk_map(req: ClassRiskMapRequest):
    if model_status != "ready":
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        datetime.strptime(req.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    results = []
    lat_range = np.arange(-90, 90, 1.0)
    lon_range = np.arange(-180, 180, 1.0)

    futures = []

    def process_coordinate(lat, lon):
        try:
            df_input = preprocess_input(lat, lon, req.date)
            X_scaled = scaler.transform(df_input)
            rnn_embedding = get_rnn_embedding(lat, lon, req.date)
            input_combined = np.hstack([X_scaled, rnn_embedding.reshape(1, -1)])
            proba = fusion_model.predict_proba(input_combined)[0]
            selected_prob = float(proba[req.crime_class_index])
            if selected_prob > 0.0:
                return {"latitude": lat, "longitude": lon, "probability": round(selected_prob, 4)}
        except Exception as e:
            print(f"[WARN] Skipped lat={lat}, lon={lon} due to: {e}")
        return None

    for lat in lat_range:
        for lon in lon_range:
            futures.append(executor.submit(process_coordinate, lat, lon))

    for future in as_completed(futures):
        result = future.result()
        if result:
            results.append(result)

    return results

# === Otros Endpoints ===
@app.get("/model/status")
def get_model_status():
    return {"model_status": model_status}

@app.post("/predict", response_model=PredictionOutput)
def predict_risk(input_data: PredictionInput):
    if model_status != "ready":
        raise HTTPException(status_code=503, detail="Model not ready")
    return generate_prediction(input_data, save=True)

@app.post("/predict/batch", response_model=List[PredictionOutput])
def predict_batch(batch: BatchPredictionInput):
    return [generate_prediction(entry, save=False) for entry in batch.inputs]

@app.post("/data/update")
def update_data():
    return {"status": "update endpoint not implemented yet"}

@app.post("/predict/map", response_model=List[PredictionOutput])
def predict_map(date: str = Query(..., description="Formato YYYY-MM-DD")):
    try:
        # Paso 1: Generar grilla
        grid_points = generate_grid(33.64, 33.92, -84.55, -84.29, step=0.01)

        # Paso 2: Filtrar por zona urbana
        filtered_points = [(lat, lon) for lat, lon in grid_points if is_in_urban_area(lat, lon)]

        # Paso 3: Predecir para cada punto
        results = []
        for lat, lon in filtered_points:
            input_data = PredictionInput(latitude=lat, longitude=lon, date=date)
            try:
                result = generate_prediction(input_data, save=False)
                if result.risk_score > 0.0:  # Si hay algún riesgo
                    results.append(result)
            except Exception as e:
                continue  # Ignora errores puntuales

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/predict/class-risk-map/threshold")
def class_risk_map_with_threshold(
    req: ClassRiskMapRequest,
    threshold: float = Query(0.2, ge=0.0, le=1.0, description="Minimum probability threshold"),
    urban_only: bool = Query(True, description="Limit to urban area only")
):
    if model_status != "ready":
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        datetime.strptime(req.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Paso 1: Generar grilla geográfica
    grid_points = generate_grid(33.64, 33.92, -84.55, -84.29, step=0.01)

    # Paso 2: Filtrar por zona urbana si se indica
    if urban_only:
        grid_points = [(lat, lon) for lat, lon in grid_points if is_in_urban_area(lat, lon)]

    # Paso 3: Paralelizar predicciones
    results = []
    futures = []

    def process_point(lat, lon):
        try:
            df_input = preprocess_input(lat, lon, req.date)
            X_scaled = scaler.transform(df_input)
            rnn_embedding = get_rnn_embedding(lat, lon, req.date)
            input_combined = np.hstack([X_scaled, rnn_embedding.reshape(1, -1)])
            proba = fusion_model.predict_proba(input_combined)[0]
            selected_prob = float(proba[req.crime_class_index])
            if selected_prob > threshold:
                return {
                    "latitude": lat,
                    "longitude": lon,
                    "probability": round(selected_prob, 4)
                }
        except Exception as e:
            print(f"[WARN] Skipped ({lat}, {lon}) due to: {e}")
        return None

    for lat, lon in grid_points:
        futures.append(executor.submit(process_point, lat, lon))

    for future in as_completed(futures):
        result = future.result()
        if result:
            results.append(result)

    # Orden opcional: de mayor a menor probabilidad
    results.sort(key=lambda x: -x["probability"])
    return results

