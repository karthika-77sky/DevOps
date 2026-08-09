import pandas as pd
import os

def preprocess_data(input_path="data/processed/ingested.csv", output_path="data/processed/preprocessed.csv"):
    df = pd.read_csv(input_path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    before = df.shape[0]
    df = df.drop_duplicates()
    print(f"Dropped {before - df.shape[0]} duplicate rows")

    missing = df.isnull().sum().sum()
    print(f"Missing values: {missing}")
    if missing > 0:
        df = df.fillna(df.median(numeric_only=True))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    preprocess_data()