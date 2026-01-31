import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# ==================================================
# Helper function: safer access to environment vars
# ==================================================
def get_env(name, default=None, required=False):
    """
    Load environment variable.
    If required=True, raises error if missing.
    If default provided, returns it when variable missing.
    """
    value = os.getenv(name, default)

    if required and value is None:
        raise ValueError(f"Missing required environment variable: {name}")

    return value


# ==================================================
# PROJECT CONFIG
# ==================================================
PROJECT_NAME = get_env("PROJECT_NAME", required=True)
ENVIRONMENT = get_env("ENVIRONMENT", "development")
RANDOM_STATE = int(get_env("RANDOM_STATE", 42))


# ==================================================
# DATA CONFIG
# ==================================================
RAW_DATA_PATH = get_env("RAW_DATA_PATH", required=True)
PROCESSED_DATA_PATH = get_env("PROCESSED_DATA_PATH")
FEATURES_DATA_PATH = get_env("FEATURES_DATA_PATH")
TARGET_COLUMN = get_env("TARGET_COLUMN", "Target")


# ==================================================
# MLFLOW CONFIGURATION
# ==================================================
MLFLOW_TRACKING_URI = get_env("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = get_env("MLFLOW_EXPERIMENT_NAME", "TFM_Experiments")
MLFLOW_ARTIFACT_URI = get_env("MLFLOW_ARTIFACT_URI", "mlruns")


# ==================================================
# DVC REMOTE CONFIGURATION
# ==================================================
DVC_REMOTE = get_env("DVC_REMOTE", "gdrive_remote")
GDRIVE_FOLDER_ID = get_env("GDRIVE_FOLDER_ID")  # Optional for now


# ==================================================
# MODEL STORAGE CONFIG
# ==================================================
MODEL_DIR = get_env("MODEL_DIR", "models")
MODEL_NAME = get_env("MODEL_NAME")   # can be empty for now


# ==================================================
# CLASS IMBALANCE / SMOTE CONFIG
# ==================================================
USE_SMOTE = get_env("USE_SMOTE", "False").lower() == "true"
SMOTE_SAMPLING_STRATEGY = get_env("SMOTE_SAMPLING_STRATEGY")


# ==================================================
# PRINT CONFIG FOR DEBUGGING
# ==================================================
def print_config():
    print("========== CONFIGURATION LOADED ==========")
    print(f"Project: {PROJECT_NAME}")
    print(f"Environment: {ENVIRONMENT}")
    print(f"Random State: {RANDOM_STATE}")
    print(f"Raw Data Path: {RAW_DATA_PATH}")
    print(f"Processed Data Path: {PROCESSED_DATA_PATH}")
    print(f"MLflow URI: {MLFLOW_TRACKING_URI}")
    print(f"DVC Remote: {DVC_REMOTE}")
    print(f"Use SMOTE: {USE_SMOTE}")
    print("==========================================")
