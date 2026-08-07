from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/best_model.pkl")

app = FastAPI(title="Heart Disease Prediction API")

# Input schema
class HeartData(BaseModel):
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
    return {"message": "Heart Disease Prediction API is running!"}

@app.post("/predict")
def predict(data: HeartData):

    input_df = pd.DataFrame([data.dict()])

    prediction = model.predict(input_df)[0]

    return {
        "prediction": int(prediction)
    }