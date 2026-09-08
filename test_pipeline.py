import os
import joblib
import pandas as pd
import numpy as np
import pytest
import yaml

def test_config_exists():
    assert os.path.exists("config/config.yaml"), "Config file does not exist!"

def test_artifacts_exist():
    # Check if model, scaler, and imputer exist locally or via dvc
    assert os.path.exists("final_model.pkl"), "Model file missing!"
    assert os.path.exists("scaler.pkl"), "Scaler file missing!"
    assert os.path.exists("imputer.pkl"), "Imputer file missing!"

def test_load_artifacts():
    model = joblib.load("final_model.pkl")
    scaler = joblib.load("scaler.pkl")
    imputer = joblib.load("imputer.pkl")
    
    assert model is not None, "Model failed to load!"
    assert scaler is not None, "Scaler failed to load!"
    assert imputer is not None, "Imputer failed to load!"

def test_pipeline_prediction_shape():
    test_path = "test.parquet"
    assert os.path.exists(test_path), "Test parquet file missing!"
    
    df_test = pd.read_parquet(test_path)
    model = joblib.load("final_model.pkl")
    scaler = joblib.load("scaler.pkl")
    imputer = joblib.load("imputer.pkl")
    
    # Process test features
    X_test = df_test.iloc[:, :-1].select_dtypes(include=["number"]).iloc[:, :4]
    X_test_imputed = imputer.transform(X_test.values)
    X_test_scaled = scaler.transform(X_test_imputed)
    X_test_model_input = np.hstack([X_test_scaled, X_test_scaled[:, :2]])
    
    predictions = model.predict(X_test_model_input)
    
    # Assertions
    assert len(predictions) == len(df_test), "Predictions count does not match test data length!"
    assert isinstance(predictions, np.ndarray), "Predictions output should be a numpy array!"