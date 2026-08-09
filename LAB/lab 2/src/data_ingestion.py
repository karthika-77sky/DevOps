import pandas as pd
import os

def ingest_data(input_path="data/raw/BostonHousing.csv", output_path="data/processed/ingested.csv"):
    df = pd.read_csv(input_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Raw data shape:", df.shape)
    return df

if __name__ == "__main__":
    ingest_data()