# app/services/utils.py

import os
import joblib
import logging
import pandas as pd

# --------------------------
# Logging Configuration
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------
# Model Persistence
# --------------------------

def save_model(model, path: str):
    """
    Saves a model to disk using joblib.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

def load_model(path: str):
    """
    Loads a model from disk using joblib.
    """
    try:
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model
    except FileNotFoundError:
        logger.warning(f"Model file not found at {path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
    return None

# --------------------------
# Data Loading Utilities
# --------------------------

def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded from {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
    return pd.DataFrame()
