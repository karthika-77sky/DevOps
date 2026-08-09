import os
import joblib
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "BostonHousing.csv",
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "model.pkl",
)

MIN_R2 = 0.70


def test_model_r2():
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

    print(f"\nModel R2: {r2:.4f}")
    print(f"Required R2: {MIN_R2:.2f}")

    assert r2 >= MIN_R2, (
        f"Model R2 {r2:.4f} is below required threshold {MIN_R2:.2f}"
    )