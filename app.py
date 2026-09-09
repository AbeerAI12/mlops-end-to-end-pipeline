from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="MLOps Inference API", version="1.0.0")

# Load artifacts at startup
try:
    model = joblib.load("final_model.pkl")
    scaler = joblib.load("scaler.pkl")
    imputer = joblib.load("imputer.pkl")
except Exception as e:
    model = scaler = imputer = None

class OrderInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "MLOps API"}

@app.get("/model/info")
def model_info():
    if model is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded.")
    return {
        "model_type": type(model).__name__,
        "version": "1.0.0",
        "status": "active"
    }

@app.post("/predict")
def predict_order(data: OrderInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded.")
    try:
        # Prepare input data
        input_data = np.array([[data.feature1, data.feature2, data.feature3, data.feature4]])
        imputed = imputer.transform(input_data)
        scaled = scaler.transform(imputed)
        model_input = np.hstack([scaled, scaled[:, :2]])
        
        prediction = model.predict(model_input)
        
        return {
            "prediction": int(prediction[0]),
            "model_version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))