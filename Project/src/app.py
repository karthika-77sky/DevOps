
from fastapi import FastAPI
import mlflow
import mlflow.pyfunc
import numpy as np
from pathlib import Path


app = FastAPI(title="Iris Classification API")


# MLflow configuration
PROJECT_DIR = Path(__file__).resolve().parent.parent
MLFLOW_DB = PROJECT_DIR / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")


# Load model from MLflow Model Registry
MODEL_URI = "models:/iris-best-model/latest"

model = mlflow.pyfunc.load_model(MODEL_URI)


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
