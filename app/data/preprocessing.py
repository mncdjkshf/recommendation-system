# app/data/preprocessing.py

import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------
# Collaborative Filtering Prep
# -------------------------------

def encode_user_item_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode user_id and item_id into numeric labels.
    Useful for matrix factorization-based models.
    """
    df = df.copy()
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df['user_idx'] = user_encoder.fit_transform(df['user_id'])
    df['item_idx'] = item_encoder.fit_transform(df['item_id'])

    logger.info("Encoded user_id and item_id into indices.")
    return df, user_encoder, item_encoder


# -------------------------------
# Content-Based Filtering Prep
# -------------------------------

def vectorize_item_descriptions(items_df: pd.DataFrame, max_features: int = 1000):
    """
    Apply TF-IDF vectorization to item descriptions.
    Returns feature matrix and vectorizer.
    """
    tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)
    try:
        tfidf_matrix = tfidf.fit_transform(items_df['description'].fillna(""))
        logger.info(f"TF-IDF vectorization complete. Shape: {tfidf_matrix.shape}")
        return tfidf_matrix, tfidf
    except Exception as e:
        logger.error(f"TF-IDF vectorization failed: {e}")
        return None, None


# -------------------------------
# Hybrid Filtering Prep
# -------------------------------

def merge_datasets(ratings_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join ratings and item metadata on 'item_id' for hybrid models.
    """
    merged_df = pd.merge(ratings_df, items_df, on='item_id', how='inner')
    logger.info(f"Merged ratings and items: {merged_df.shape}")
    return merged_df

