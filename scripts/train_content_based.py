# app/scripts/train_content_based.py

import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

from app.data.load_data import load_items_data
from app.data.preprocessing import vectorize_item_descriptions
from app.config import TFIDF_MATRIX_FILE, TFIDF_VECTORIZER_FILE

def train_content_based_model():
    """
    Train content-based filtering model using TF-IDF + cosine similarity.
    """
    # Load item metadata
    items_df = load_items_data()

    # TF-IDF vectorization of item descriptions
    tfidf_matrix, tfidf_vectorizer = vectorize_item_descriptions(items_df)

    if tfidf_matrix is None:
        raise ValueError("TF-IDF vectorization failed. Check item descriptions.")

    # Save TF-IDF vectorizer
    joblib.dump(tfidf_vectorizer, TFIDF_VECTORIZER_FILE)
    print(f"✅ Saved TF-IDF vectorizer to {TFIDF_VECTORIZER_FILE}")

    # Save TF-IDF matrix (sparse format)
    sparse.save_npz(TFIDF_MATRIX_FILE, tfidf_matrix)
    print(f"✅ Saved TF-IDF matrix to {TFIDF_MATRIX_FILE}")

    # Optionally: Compute similarity matrix (not saved here to save space)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print("✅ Cosine similarity matrix computed (not saved).")

    return similarity_matrix

def main():
    print("📌 Training content-based model...")
    similarity_matrix = train_content_based_model()
    print("✅ Content-based model training complete.")

if __name__ == "__main__":
    main()
