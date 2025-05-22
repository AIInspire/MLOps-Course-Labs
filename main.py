from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Churn Prediction API with Prometheus")

model = joblib.load("model.pkl")

class CustomerData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: CustomerData):
    input_df = [data.dict()]
    prediction = model.predict(input_df)
    return {"prediction": int(prediction[0])}

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

"""
🎯 Why These Metrics?
We used prometheus-fastapi-instrumentator to monitor:

1. Request Count: How many times each endpoint is hit.
2. Request Latency: How long requests take. Crucial for spotting bottlenecks.
3. Status Codes: To track errors (e.g., 500, 404).
4. Model Prediction Endpoint Metrics: To ensure predictions stay performant under load.
"""