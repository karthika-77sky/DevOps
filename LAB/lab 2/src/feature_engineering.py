import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2

def engineer_features(input_path="data/processed/preprocessed.csv", output_dir="data/processed"):
    df = pd.read_csv(input_path)
    X = df.drop(columns=["medv"])
    y = df["medv"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    os.makedirs(output_dir, exist_ok=True)
    X_train_scaled.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test_scaled.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print("Train shape:", X_train_scaled.shape)
    print("Test shape :", X_test_scaled.shape)

if __name__ == "__main__":
    engineer_features()