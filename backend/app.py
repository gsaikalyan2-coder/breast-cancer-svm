# backend/app.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, numpy as np, os

# ── Load model ──────────────────────────────────────────────────────────────
MODEL_PATH = "/app/model/breast_cancer_svm.pkl"
model = joblib.load(MODEL_PATH)
print(f"✓ Model loaded from {MODEL_PATH}")

app = FastAPI(
    title       = "Breast Cancer SVM API",
    description = "Predicts malignant/benign from 30 cell nucleus measurements",
    version     = "1.0.0"
)

# ── CORS — allows Streamlit frontend to call this API ───────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# ── Input schema — all 30 features ──────────────────────────────────────────
class TumorFeatures(BaseModel):
    mean_radius:              float
    mean_texture:             float
    mean_perimeter:           float
    mean_area:                float
    mean_smoothness:          float
    mean_compactness:         float
    mean_concavity:           float
    mean_concave_points:      float
    mean_symmetry:            float
    mean_fractal_dimension:   float
    radius_error:             float
    texture_error:            float
    perimeter_error:          float
    area_error:               float
    smoothness_error:         float
    compactness_error:        float
    concavity_error:          float
    concave_points_error:     float
    symmetry_error:           float
    fractal_dimension_error:  float
    worst_radius:             float
    worst_texture:            float
    worst_perimeter:          float
    worst_area:               float
    worst_smoothness:         float
    worst_compactness:        float
    worst_concavity:          float
    worst_concave_points:     float
    worst_symmetry:           float
    worst_fractal_dimension:  float

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status":  "running",
        "model":   "breast_cancer_svm",
        "version": "1.0.0"
    }

@app.get("/health")
def health():
    return {"healthy": True}

@app.post("/predict")
def predict(features: TumorFeatures):
    values     = np.array(list(features.dict().values())).reshape(1, -1)
    prediction = model.predict(values)[0]
    proba      = model.predict_proba(values)[0]
    return {
        "prediction": int(prediction),
        "label":      "benign" if prediction == 1 else "malignant",
        "confidence": round(float(proba.max()) * 100, 2),
        "probabilities": {
            "malignant": round(float(proba[0]) * 100, 2),
            "benign":    round(float(proba[1]) * 100, 2),
        }
    }