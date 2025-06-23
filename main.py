from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model, Model
from types import SimpleNamespace

# === Cargar modelos y datos ===
fusion_model = joblib.load("exported_models/fusion_model.pkl")
scaler, feature_names = joblib.load("exported_models/scaler.pkl")
rnn_model = load_model("exported_models/rnn_model.keras")

df_contextual = pd.read_csv("df_contextual.csv", low_memory=False)
df_contextual["datetime"] = pd.to_datetime(df_contextual["datetime"], errors="coerce")

# === Preparar modelo ===
model = SimpleNamespace()
model.rnn_model = rnn_model
model.fusion_model = fusion_model
model.scaler = scaler
model.temporal_window = 30

# === Diccionario de clases (resumido) ===
crime_class_dict = {
    0: "Liquor Law Violations", 1: "Impersonation", 2: "All Other Offenses",
    16: "Theft From Motor Vehicle", 21: "Motor Vehicle Theft", 53: "Incest"
    # Puedes agregar los 57 si quieres
}

# === FastAPI app ===
app = FastAPI(title="Predicción de Crímenes", version="1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # URL de tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Esquema de respuesta ===
class CrimePrediction(BaseModel):
    date: datetime
    spatial_cluster: int
    latitude: float
    longitude: float
    class_id: int
    crime_type: str
    probability: float


# === Función de predicción ===
def predict_top_crimes_for_date_future(model, df_contextual, target_datetime, top_n=5):
    from math import sin, cos, pi

    def compute_temporal_features(dt):
        hour = dt.hour
        day = dt.day
        month = dt.month
        return {
            "hour_sin": sin(2 * pi * hour / 24),
            "hour_cos": cos(2 * pi * hour / 24),
            "day_sin": sin(2 * pi * day / 31),
            "day_cos": cos(2 * pi * day / 31),
            "month_sin": sin(2 * pi * month / 12),
            "month_cos": cos(2 * pi * month / 12),
            "hour": hour,
            "day_of_week": dt.weekday(),
            "month": month,
            "day_of_year": dt.timetuple().tm_yday,
            "year": dt.year,
            "quarter": (month - 1) // 3 + 1,
        }

    all_predictions = []
    clusters = df_contextual["spatial_cluster"].unique()

    for cluster_id in clusters:
        cluster_df = df_contextual[df_contextual["spatial_cluster"] == cluster_id].sort_values("datetime")
        if len(cluster_df) < model.temporal_window:
            continue

        recent_seq = cluster_df.tail(model.temporal_window)
        temporal_features = ["crime_count_7d", "crime_count_30d", "temporal_weight"]

        try:
            sequence = recent_seq[temporal_features].values[np.newaxis, :, :]
            embedding_model = Model(inputs=model.rnn_model.input,
                                    outputs=model.rnn_model.get_layer("dense_embedding").output)
            embedding = embedding_model.predict(sequence, verbose=0)

            context_vector = pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)
            for col in feature_names:
                if col in recent_seq.columns:
                    val = pd.to_numeric(recent_seq[col], errors='coerce').mean()
                    if pd.notna(val):
                        context_vector.at[0, col] = val

            temporal_injection = compute_temporal_features(target_datetime)
            for key, val in temporal_injection.items():
                if key in context_vector.columns:
                    context_vector.at[0, key] = val

            scaled_context = model.scaler.transform(context_vector)
            combined_input = np.hstack([scaled_context, embedding])
            probs = model.fusion_model.predict_proba(combined_input)[0]

            for class_id, prob in enumerate(probs):
                all_predictions.append({
                    "date": target_datetime,
                    "spatial_cluster": cluster_id,
                    "latitude": recent_seq["Latitude"].mean(),
                    "longitude": recent_seq["Longitude"].mean(),
                    "class_id": class_id,
                    "crime_type": crime_class_dict.get(class_id, f"Class {class_id}"),
                    "probability": float(prob),
                })

        except Exception as e:
            continue

    result_df = pd.DataFrame(all_predictions)
    top_by_class = result_df.sort_values("probability", ascending=False).drop_duplicates("class_id")
    return top_by_class.sort_values("probability", ascending=False).head(top_n)


# === Ruta de predicción ===
@app.get("/predict_crimes", response_model=List[CrimePrediction])
def predict_crimes(
    datetime_str: str = Query(..., example="2025-09-15T20:00:00"),
    top_n: int = Query(5, ge=1, le=50)
):
    try:
        target_datetime = datetime.fromisoformat(datetime_str)
    except ValueError:
        return []

    result = predict_top_crimes_for_date_future(model, df_contextual, target_datetime, top_n)
    return result.to_dict(orient="records")


