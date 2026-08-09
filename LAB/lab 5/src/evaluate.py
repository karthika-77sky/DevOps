import json
import os

import joblib
import pandas as pd

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


def evaluate_model():
    data = pd.read_csv(DATA_PATH)

    X = data.drop("medv", axis=1)
    y = data["medv"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)

    with open(METRICS_PATH) as file:
        metrics = json.load(file)

    minimum_r2 = metrics["minimum_r2"]

    print(f"R2_SCORE={r2:.4f}")
    print(f"MINIMUM_R2={minimum_r2:.2f}")

    if r2 < minimum_r2:
        raise ValueError(
            f"Quality gate failed: R2 {r2:.4f} < {minimum_r2:.2f}"
        )

    print("QUALITY_GATE=PASSED")


if __name__ == "__main__":
    evaluate_model()