# app/models/content_based.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedFiltering:
    def __init__(self, item_df: pd.DataFrame, text_column: str = "description"):
        """
        item_df: DataFrame with at least [item_id, <text_column>]
        text_column: column to base similarity on (e.g., title, tags, genres, or description)
        """
        self.item_df = item_df
        self.text_column = text_column
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.item_index_map = {}
        self._prepare()

    def _prepare(self):
        # Fill NA with empty string to prevent TF-IDF errors
        self.item_df[self.text_column] = self.item_df[self.text_column].fillna("")

        # Build TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = vectorizer.fit_transform(self.item_df[self.text_column])

        # Compute cosine similarity between all items
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

        # Map item_id to index in similarity matrix
        self.item_index_map = {
            item_id: idx for idx, item_id in enumerate(self.item_df["item_id"])
        }

    def recommend_similar_items(self, item_id: int, top_n: int = 10) -> list:
        """
        Recommend similar items based on content.
        Returns list of (item_id, similarity_score).
        """
        if item_id not in self.item_index_map:
            return []

        idx = self.item_index_map[item_id]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))

        # Sort by similarity score, exclude the item itself
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_matches = [(self.item_df["item_id"].iloc[i], score)
                       for i, score in sim_scores if self.item_df["item_id"].iloc[i] != item_id]

        return top_matches[:top_n]
