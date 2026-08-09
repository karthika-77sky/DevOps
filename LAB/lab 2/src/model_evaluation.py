import pandas as pd
import joblib
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(data_dir="data/processed", model_path="models/model.pkl", metrics_path="metrics.json"):
    model = joblib.load(model_path)
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").squeeze()

    y_pred = model.predict(X_test)

    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_test, y_pred),
        "r2_score": r2_score(y_test, y_pred),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    evaluate_model()