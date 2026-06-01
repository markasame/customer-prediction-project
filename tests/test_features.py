"""
tests/test_features.py
Unit tests for the feature engineering pipeline.
Run: pytest tests/
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features.build import clean_raw_data, build_preprocessing_pipeline


@pytest.fixture
def sample_cfg():
    return {
        "data": {"target_column": "Churn"},
        "features": {
            "numeric": ["tenure", "MonthlyCharges", "TotalCharges"],
            "categorical": ["gender", "Contract"],
            "drop": ["customerID"],
        },
    }


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "customerID": ["A1", "A2", "A3"],
        "gender": ["Male", "Female", "Male"],
        "tenure": [12, 24, 0],
        "MonthlyCharges": [65.0, 80.0, 50.0],
        "TotalCharges": ["780", "1920", " "],  # space = new customer
        "Contract": ["Month-to-month", "Two year", "One year"],
        "Churn": ["Yes", "No", "No"],
    })


def test_clean_raw_data_removes_customer_id(sample_df, sample_cfg):
    X, y = clean_raw_data(sample_df, sample_cfg)
    assert "customerID" not in X.columns


def test_clean_raw_data_encodes_target(sample_df, sample_cfg):
    X, y = clean_raw_data(sample_df, sample_cfg)
    assert set(y.unique()).issubset({0, 1})
    assert y.iloc[0] == 1  # "Yes" → 1
    assert y.iloc[1] == 0  # "No" → 0


def test_clean_raw_data_handles_empty_total_charges(sample_df, sample_cfg):
    X, y = clean_raw_data(sample_df, sample_cfg)
    # New customer (TotalCharges was " ") should be imputed with MonthlyCharges
    assert not X["TotalCharges"].isnull().any()
    assert X["TotalCharges"].iloc[2] == 50.0


def test_preprocessing_pipeline_output_shape(sample_df, sample_cfg):
    X, y = clean_raw_data(sample_df, sample_cfg)
    preprocessor = build_preprocessing_pipeline(sample_cfg)
    X_processed = preprocessor.fit_transform(X)
    expected_cols = len(sample_cfg["features"]["numeric"]) + len(sample_cfg["features"]["categorical"])
    assert X_processed.shape == (len(sample_df), expected_cols)


def test_preprocessing_handles_unseen_categories(sample_df, sample_cfg):
    X, y = clean_raw_data(sample_df, sample_cfg)
    preprocessor = build_preprocessing_pipeline(sample_cfg)
    preprocessor.fit_transform(X)

    # New data with an unseen contract type
    new_data = pd.DataFrame({
        "gender": ["Female"],
        "tenure": [6.0],
        "MonthlyCharges": [55.0],
        "TotalCharges": [330.0],
        "Contract": ["Quarterly"],  # unseen category
    })
    result = preprocessor.transform(new_data)
    assert result is not None
    assert result.shape[1] == 5
