from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

MODEL_PATH = "model.pkl"

app = FastAPI(title="SberAutoSubscription Predictor")

model = None


class VisitInput(BaseModel):
    data: dict


@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {"status": "ok", "message": "Model API is running"}


@app.post("/predict")
def predict(payload: VisitInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    try:
        X = pd.DataFrame([payload.data])
        proba = model.predict_proba(X)[:, 1][0]
        pred = int(proba >= 0.5)
        return {"prediction": pred, "probability": float(proba)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
