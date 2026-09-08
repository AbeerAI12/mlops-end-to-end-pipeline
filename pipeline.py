import yaml
import joblib
import pandas as pd
import numpy as np
import os
import logging
import mlflow
import mlflow.sklearn

# Setup logging configuration (console and file handlers)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pipeline.log")
    ]
)
logger = logging.getLogger("MLOps_Pipeline")

def load_config(config_path="config/config.yaml"):
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

def load_artifacts():
    try:
        config = load_config()
        model_path = config["paths"]["model_path"]
        scaler_path = config["paths"]["scaler_path"]
        imputer_path = config["paths"]["imputer_path"]
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        imputer = joblib.load(imputer_path)
        
        logger.info("All model artifacts (model, scaler, imputer) loaded successfully!")
        return model, scaler, imputer
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        raise

def run_pipeline():
    try:
        logger.info("Starting MLOps pipeline execution with MLflow tracking...")
        
        # Start MLflow run for experiment tracking
        with mlflow.start_run(run_name="Inference_Batch_Run"):
            
            # 1. Load artifacts and configuration
            model, scaler, imputer = load_artifacts()
            config = load_config()
            
            # Log parameters to MLflow
            mlflow.log_param("model_type", type(model).__name__)
            test_data_path = config["paths"].get("test_path", "test.parquet")
            mlflow.log_param("test_data_source", test_data_path)
            
            # 2. Read test data with error handling
            if not os.path.exists(test_data_path):
                raise FileNotFoundError(f"Test data file not found at: {test_data_path}")
                
            df_test = pd.read_parquet(test_data_path)
            logger.info(f"Test data loaded successfully with shape: {df_test.shape}")
            
            # 3. Feature selection and transformation
            X_test = df_test.iloc[:, :-1].select_dtypes(include=["number"]).iloc[:, :4]
            X_test_imputed = imputer.transform(X_test.values)
            X_test_scaled = scaler.transform(X_test_imputed)
            
            # Dimension matching for the model requirement
            X_test_model_input = np.hstack([X_test_scaled, X_test_scaled[:, :2]])
            
            # 4. Generate predictions
            predictions = model.predict(X_test_model_input)
            total_predictions = len(predictions)
            logger.info(f"Predictions generated successfully! Total predictions: {total_predictions}")
            
            # Log metrics to MLflow
            mlflow.log_metric("total_predictions_count", total_predictions)
            
            # 5. Save results and log artifact
            os.makedirs("outputs", exist_ok=True)
            output_df = pd.DataFrame({"Prediction": predictions})
            output_path = "outputs/predictions.csv"
            output_df.to_csv(output_path, index=False)
            
            # Log the output file as an artifact in MLflow
            mlflow.log_artifact(output_path)
            logger.info(f"Predictions successfully saved and logged to MLflow from {output_path}")
            
        logger.info("Pipeline execution and MLflow tracking completed successfully!")
        
    except Exception as e:
        logger.critical(f"Pipeline failed due to an error: {e}", exc_info=True)

if __name__ == "__main__":
    run_pipeline()