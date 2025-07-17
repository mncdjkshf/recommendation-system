# app/data/load_data.py

import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the directory where CSVs are stored
DATA_DIR = os.path.dirname(__file__)

def load_ratings_data(file_name: str = "ratings.csv") -> pd.DataFrame:
    """
    Load user-item ratings data.
    Expected columns: user_id, item_id, rating
    """
    path = os.path.join(DATA_DIR, file_name)
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded ratings data from {path}, shape={df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Ratings file not found: {path}")
    except Exception as e:
        logger.error(f"Error loading ratings data: {e}")
    return pd.DataFrame()


def load_items_data(file_name: str = "items.csv") -> pd.DataFrame:
    """
    Load item metadata.
    Expected columns: item_id, description
    """
    path = os.path.join(DATA_DIR, file_name)
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded items data from {path}, shape={df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Items file not found: {path}")
    except Exception as e:
        logger.error(f"Error loading items data: {e}")
    return pd.DataFrame()


def clean_ratings_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean missing or invalid ratings.
    """
    initial_shape = df.shape
    df = df.dropna(subset=["user_id", "item_id", "rating"])
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df = df[df["rating"].between(0.0, 5.0)]
    logger.info(f"Cleaned ratings data: {initial_shape} -> {df.shape}")
    return df


def clean_items_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean item metadata.
    """
    initial_shape = df.shape
    df = df.dropna(subset=["item_id", "description"])
    logger.info(f"Cleaned items data: {initial_shape} -> {df.shape}")
    return df
