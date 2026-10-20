import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Heart Disease Risk Prediction API")

Instrumentator().instrument(app).expose(app)

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.get("/")
def home():
    return {"status": "API is online"}

@app.post("/predict")
def predict(data: PatientData):
    features = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]])

    features_scaled = scaler.transform(features)
    prediction = int(model.predict(features_scaled)[0])
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = float(np.max(probabilities))

    return {
        "heart_disease_risk": prediction,
        "confidence": confidence,
        "probability_distribution": {
            "no_risk": float(probabilities[0]),
            "high_risk": float(probabilities[1])
        }
    }