"""
src/api/main.py
FastAPI REST API serving the churn prediction model.

Run:  uvicorn src.api.main:app --reload
Docs: http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import time
import logging
from src.models.predict import load_artifacts, predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability using an XGBoost model.",
    version="1.0.0",
)

# Load model once at startup
model, pipeline, cfg = load_artifacts()
prediction_count = 0
start_time = time.time()


class CustomerFeatures(BaseModel):
    """Input schema — mirrors the Telco churn dataset fields."""
    gender: str = Field(example="Male")
    SeniorCitizen: int = Field(ge=0, le=1, example=0)
    Partner: str = Field(example="Yes")
    Dependents: str = Field(example="No")
    tenure: float = Field(ge=0, example=24)
    PhoneService: str = Field(example="Yes")
    MultipleLines: str = Field(example="No")
    InternetService: str = Field(example="Fiber optic")
    OnlineSecurity: str = Field(example="No")
    OnlineBackup: str = Field(example="Yes")
    DeviceProtection: str = Field(example="No")
    TechSupport: str = Field(example="No")
    StreamingTV: str = Field(example="Yes")
    StreamingMovies: str = Field(example="Yes")
    Contract: str = Field(example="Month-to-month")
    PaperlessBilling: str = Field(example="Yes")
    PaymentMethod: str = Field(example="Electronic check")
    MonthlyCharges: float = Field(ge=0, example=85.0)
    TotalCharges: float = Field(ge=0, example=2040.0)


class PredictionResponse(BaseModel):
    churn_probability: float
    risk_label: str  # Low / Medium / High
    top_risk_factors: list[str]
    model_version: str = "1.0.0"


@app.get("/health")
def health():
    return {"status": "ok", "uptime_seconds": round(time.time() - start_time)}


@app.get("/metrics")
def metrics():
    return {
        "total_predictions": prediction_count,
        "uptime_seconds": round(time.time() - start_time),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerFeatures):
    global prediction_count
    try:
        result = predict(customer.model_dump(), model, pipeline, cfg)
        prediction_count += 1
        logger.info(f"Prediction: prob={result['churn_probability']}, risk={result['risk_label']}")
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(customers: list[CustomerFeatures]):
    """Predict for multiple customers in one request."""
    global prediction_count
    results = []
    for customer in customers:
        try:
            result = predict(customer.model_dump(), model, pipeline, cfg)
            results.append(PredictionResponse(**result))
            prediction_count += 1
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return results
