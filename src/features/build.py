"""
src/features/build.py
Builds the full preprocessing pipeline: cleaning, encoding, scaling, SMOTE.
Run: python src/features/build.py
"""
import os
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

CONFIG_PATH = "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def clean_raw_data(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Clean and split raw DataFrame into features and target."""
    df = df.copy()

    # Drop useless columns
    df.drop(columns=cfg["features"]["drop"], errors="ignore", inplace=True)

    # Fix TotalCharges: it's stored as string with spaces for new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

    # Encode target
    target_col = cfg["data"]["target_column"]
    y = (df[target_col].str.strip() == "Yes").astype(int)
    X = df.drop(columns=[target_col])

    return X, y


def build_preprocessing_pipeline(cfg: dict) -> ColumnTransformer:
    """Build sklearn ColumnTransformer for numeric + categorical features."""
    numeric_features = cfg["features"]["numeric"]
    categorical_features = cfg["features"]["categorical"]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def save_features(X_resampled, y_resampled, path: str, cfg: dict):
    """Save processed feature matrix as CSV for reference/monitoring."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    numeric_features = cfg["features"]["numeric"]
    categorical_features = cfg["features"]["categorical"]
    all_features = numeric_features + categorical_features

    df_out = pd.DataFrame(X_resampled, columns=all_features)
    df_out["Churn"] = y_resampled
    df_out.to_csv(path, index=False)
    print(f"Processed features saved to {path} ({len(df_out)} rows after SMOTE)")


def main():
    cfg = load_config()

    # Load raw data
    df = pd.read_csv(cfg["data"]["raw_path"])
    X, y = clean_raw_data(df, cfg)
    print(f"Raw features: {X.shape}, Churn rate: {y.mean():.1%}")

    # Build & fit preprocessor
    preprocessor = build_preprocessing_pipeline(cfg)
    X_processed = preprocessor.fit_transform(X)
    print(f"After preprocessing: {X_processed.shape}")

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=cfg["model"]["random_state"])
    X_resampled, y_resampled = smote.fit_resample(X_processed, y)
    print(f"After SMOTE: {X_resampled.shape}, Churn rate: {y_resampled.mean():.1%}")

    # Save pipeline artifact and processed features
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, cfg["model"]["pipeline_path"])
    print(f"Pipeline saved to {cfg['model']['pipeline_path']}")

    save_features(X_resampled, y_resampled, cfg["data"]["processed_path"], cfg)
    print("\nFeature engineering complete. Ready to train.")


if __name__ == "__main__":
    main()
