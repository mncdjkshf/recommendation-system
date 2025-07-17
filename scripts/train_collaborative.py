# app/scripts/train_collaborative.py

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import joblib
import os

from app.data.load_data import load_ratings_data, clean_ratings_data
from app.config import RATINGS_FILE, COLLAB_MODEL_FILE, NUM_FACTORS, NUM_EPOCHS, LEARNING_RATE

def prepare_dataset_for_surprise(ratings_df: pd.DataFrame):
    """
    Converts a DataFrame into a Surprise-compatible dataset
    """
    reader = Reader(rating_scale=(0.0, 5.0))
    data = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)
    return data

def train_svd_model(ratings_df: pd.DataFrame):
    """
    Train an SVD (matrix factorization) model using Surprise.
    """
    # Convert data to Surprise format
    data = prepare_dataset_for_surprise(ratings_df)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # Define the SVD model
    algo = SVD(n_factors=NUM_FACTORS, n_epochs=NUM_EPOCHS, lr_all=LEARNING_RATE)

    # Train the model
    print("Training collaborative filtering model...")
    algo.fit(trainset)

    # Evaluate
    predictions = algo.test(testset)
    from surprise import accuracy
    rmse = accuracy.rmse(predictions)
    print(f"✅ RMSE on test set: {rmse:.4f}")

    return algo

def save_model(model, model_path: str):
    """
    Save the trained collaborative filtering model
    """
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")

def main():
    # Load and clean ratings data
    ratings_df = load_ratings_data()
    ratings_df = clean_ratings_data(ratings_df)

    # Train the model
    model = train_svd_model(ratings_df)

    # Save the model
    save_model(model, COLLAB_MODEL_FILE)

if __name__
