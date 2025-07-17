# app/scripts/evaluate_models.py

import pandas as pd
import joblib
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy import sparse

from app.config import (
    RATINGS_FILE,
    ITEMS_FILE,
    COLLAB_MODEL_FILE,
    TFIDF_MATRIX_FILE,
    TFIDF_VECTORIZER_FILE
)
from app.data.load_data import load_ratings_data, load_items_data


def evaluate_collaborative():
    print("\n📌 Evaluating Collaborative Filtering Model...")

    # Load model
    model = joblib.load(COLLAB_MODEL_FILE)

    # Load and prepare data
    ratings_df = load_ratings_data()
    reader = Reader(rating_scale=(0.0, 5.0))
    data = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # Evaluate
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    print(f"✅ Collaborative Filtering RMSE: {rmse:.4f}, MAE: {mae:.4f}")


def evaluate_content_based(top_n=5):
    print("\n📌 Evaluating Content-Based Filtering Model...")

    # Load data and TF-IDF model
    items_df = load_items_data()
    tfidf_matrix = sparse.load_npz(TFIDF_MATRIX_FILE)
    tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_FILE)

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Recommend for each item and calculate precision-like metric
    correct = 0
    total = 0

    for idx, row in items_df.iterrows():
        item_idx = idx
        sim_scores = list(enumerate(similarity_matrix[item_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_similar_items = [i[0] for i in sim_scores[1:top_n + 1]]

        # Dummy metric: whether any top similar item has same category (or tag)
        target_tag = row.get("tag") or row.get("category")
        if not target_tag:
            continue

        for similar_item_idx in top_similar_items:
            similar_item_tag = items_df.iloc[similar_item_idx].get("tag") or items_df.iloc[similar_item_idx].get(
                "category")
            if similar_item_tag == target_tag:
                correct += 1
                break
        total += 1

    precision = correct / total if total > 0 else 0
    print(f"✅ Content-Based Top-{top_n} Precision (by tag/category match): {precision:.4f}")


def main():
    evaluate_collaborative()
    evaluate_content_based()


if __name__ == "__main__":
    main()
