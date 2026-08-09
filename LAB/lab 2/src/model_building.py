import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42
N_ESTIMATORS = 200
MAX_DEPTH = 6

def build_model(data_dir="data/processed", model_path="models/model.pkl"):
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze()

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print("Model trained:", model)

if __name__ == "__main__":
    build_model()