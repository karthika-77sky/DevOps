from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="Iris Classification API")


# Load the trained model
model = joblib.load("model.joblib")


@app.get("/")
def home():
    return {
        "message": "Iris Classification API is running!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(features: list[float]):
    data = np.array(features).reshape(1, -1)

    prediction = model.predict(data)[0]

    return {
        "prediction": int(prediction)
    }