import json
import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "BostonHousing.csv",
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model.pkl",
)

METRICS_PATH = os.path.join(
    BASE_DIR,
    "metrics.json",
)


def train_model():
    data = pd.read_csv(DATA_PATH)

    X = data.drop("medv", axis=1)
    y = data["medv"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)

    joblib.dump(model, MODEL_PATH)

    metrics = {
        "r2": round(r2, 4),
        "minimum_r2": 0.70,
    }

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=2)

    print(f"R2_SCORE={r2:.4f}")
    print(f"MODEL_SAVED={MODEL_PATH}")
    print(f"METRICS_SAVED={METRICS_PATH}")

    return r2


if __name__ == "__main__":
    train_model()