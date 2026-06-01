"""
src/data/ingest.py
Downloads the Telco churn dataset and validates it.
Run: python src/data/ingest.py
"""
import os
import yaml
import requests
import pandas as pd

CONFIG_PATH = "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def download_dataset(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading dataset from {url}...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(dest, "wb") as f:
        f.write(response.content)
    print(f"Saved to {dest}")


def validate_dataset(path: str, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    assert target_col in df.columns, f"Target column '{target_col}' not found!"
    print(f"\nTarget distribution:\n{df[target_col].value_counts(normalize=True).round(3)}")

    churn_rate = df[target_col].value_counts(normalize=True).get("Yes", 0)
    print(f"\nChurn rate: {churn_rate:.1%} — class imbalance detected." if churn_rate < 0.3 else f"\nChurn rate: {churn_rate:.1%}")

    return df


def main():
    cfg = load_config()
    raw_path = cfg["data"]["raw_path"]
    url = cfg["data"]["dataset_url"]
    target = cfg["data"]["target_column"]

    if not os.path.exists(raw_path):
        download_dataset(url, raw_path)
    else:
        print(f"Dataset already exists at {raw_path}, skipping download.")

    df = validate_dataset(raw_path, target)
    print(f"\nData ingestion complete. {len(df)} rows ready for feature engineering.")


if __name__ == "__main__":
    main()
