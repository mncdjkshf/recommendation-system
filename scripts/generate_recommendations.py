# app/scripts/generate_recommendations.py

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from surprise import Dataset, Reader
from surprise import SVD
from scipy import sparse

from app.config import (
    RATINGS_FILE,
    ITEMS_FILE,
    COLLAB_MODEL_FILE,
    TFIDF_MATRIX_FILE,
    TFIDF_VECTORIZER_FILE,
    TOP_N
)

from app.data.load_data import load_ratings_data, load_items_data


def get_unrated_items(user_id, ratings_df, items_df):
    rated_items = ratings_df[ratings_df['user_id'] == user_id]['item_id'].unique()
    return items_df[~items_df['item_id'].isin(rated_items)]


def recommend_collaborative(user_id, ratings_df, items_df, model, top_n=TOP_N):
    print("\n🔷 Collaborative Filtering Recommendations")
    unrated_items = get_unrated_items(user_id, ratings_df, items_df)

    predictions = [
        (item_id, model.predict(str(user_id), str(item_id)).est)
        for item_id in unrated_items['item_id']
    ]
    top_items = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
    for i, (item_id, score) in enumerate(top_items, start=1):
        title = items_df[items_df['item_id'] == item_id]['title'].values[0]
        print(f"{i}. {title} (Predicted rating: {score:.2f})")


def recommend_content_based(user_id, ratings_df, items_df, tfidf_matrix, top_n=TOP_N):
    print("\n🔷 Content-Based Filtering Recommendations")
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    if user_ratings.empty:
        print("No user history found. Content-based filtering needs at least one rated item.")
        return

    # Assume last rated item is most recent preference
    last_item_id = user_ratings.sort_values(by='timestamp', ascending=False).iloc[0]['item_id']
    last_item_idx = items_df.index[items_df['item_id'] == last_item_id].tolist()
    if not last_item_idx:
        print("Last item not found in item dataset.")
        return
    last_item_idx = last_item_idx[0]

    similarity_scores = cosine_similarity(tfidf_matrix[last_item_idx], tfidf_matrix).flatten()
    similar_indices = similarity_scores.argsort()[::-1][1:top_n + 1]

    for i, idx in enumerate(similar_indices, start=1):
        item_id = items_df.iloc[idx]['item_id']
        title = items_df.iloc[idx]['title']
        print(f"{i}. {title} (Similarity score: {similarity_scores[idx]:.2f})")


def recommend_hybrid(user_id, ratings_df, items_df, collab_model, tfidf_matrix, alpha=0.5, top_n=TOP_N):
    print("\n🔷 Hybrid Filtering Recommendations")
    unrated_items = get_unrated_items(user_id, ratings_df, items_df)

    hybrid_scores = []
    for idx, row in unrated_items.iterrows():
        item_id = row['item_id']

        # Collaborative score
        collab_score = collab_model.predict(str(user_id), str(item_id)).est

        # Content score: similarity with user's last item
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        if user_ratings.empty:
            content_score = 0
        else:
            last_item_id = user_ratings.sort_values(by='timestamp', ascending=False).iloc[0]['item_id']
            try:
                last_idx = items_df.index[items_df['item_id'] == last_item_id].tolist()[0]
                current_idx = items_df.index[items_df['item_id'] == item_id].tolist()[0]
                content_score = cosine_similarity(tfidf_matrix[last_idx], tfidf_matrix[current_idx])[0][0]
            except IndexError:
                content_score = 0

        hybrid_score = alpha * collab_score + (1 - alpha) * content_score
        hybrid_scores.append((item_id, hybrid_score))

    top_items = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:top_n]
    for i, (item_id, score) in enumerate(top_items, start=1):
        title = items_df[items_df['item_id'] == item_id]['title'].values[0]
        print(f"{i}. {title} (Hybrid score: {score:.2f})")


def main():
    user_id = input("Enter user_id to get recommendations: ").strip()

    # Load data
    ratings_df = load_ratings_data()
    items_df = load_items_data()
    tfidf_matrix = sparse.load_npz(TFIDF_MATRIX_FILE)
    collab_model = joblib.load(COLLAB_MODEL_FILE)

    # Generate recommendations
    recommend_collaborative(user_id, ratings_df, items_df, collab_model)
    recommend_content_based(user_id, ratings_df, items_df, tfidf_matrix)
    recommend_hybrid(user_id, ratings_df, items_df, collab_model, tfidf_matrix)

if __name__ == "__main__":
    main()
