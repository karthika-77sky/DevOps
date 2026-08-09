from fastapi import FastAPI
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(title="Iris Classification API")

# Load trained model
MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pkl"
model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {"message": "Iris Classification API is running!"}


@app.post("/predict")
def predict(features: list[float]):
    data = np.array(features).reshape(1, -1)

    prediction = model.predict(data)[0]

    return {
        "prediction": int(prediction)
    }