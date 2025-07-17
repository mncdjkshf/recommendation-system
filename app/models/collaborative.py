# app/models/collaborative.py

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFiltering:
    def __init__(self, ratings_df: pd.DataFrame, similarity_type: str = "user"):
        """
        ratings_df: DataFrame with columns [user_id, item_id, rating]
        similarity_type: "user" or "item"
        """
        self.ratings_df = ratings_df
        self.similarity_type = similarity_type
        self.user_item_matrix = None
        self.similarity_matrix = None
        self._prepare()

    def _prepare(self):
        # Create user-item matrix (rows: users, columns: items)
        self.user_item_matrix = self.ratings_df.pivot_table(
            index="user_id", columns="item_id", values="rating"
        ).fillna(0)

        # Compute similarity matrix
        if self.similarity_type == "user":
            self.similarity_matrix = cosine_similarity(self.user_item_matrix)
            self.sim_index = self.user_item_matrix.index
        else:  # item-based
            self.similarity_matrix = cosine_similarity(self.user_item_matrix.T)
            self.sim_index = self.user_item_matrix.columns

    def recommend_for_user(self, user_id: int, top_n: int = 10) -> list:
        """
        Recommend top-N items for a user based on collaborative filtering.
        Returns a list of (item_id, predicted_score).
        """
        if self.similarity_type == "user":
            return self._user_based_recommend(user_id, top_n)
        else:
            return self._item_based_recommend(user_id, top_n)

    def _user_based_recommend(self, user_id: int, top_n: int = 10):
        if user_id not in self.user_item_matrix.index:
            return []

        user_idx = self.user_item_matrix.index.get_loc(user_id)
        similarity_scores = self.similarity_matrix[user_idx]

        # Compute weighted sum of other users' ratings
        weighted_ratings = np.dot(similarity_scores, self.user_item_matrix.values)
        sim_sum = similarity_scores.sum()

        predicted_scores = weighted_ratings / sim_sum if sim_sum != 0 else weighted_ratings

        # Recommend items not already rated by the user
        user_rated_items = self.user_item_matrix.loc[user_id]
        unrated_items = user_rated_items[user_rated_items == 0]

        recommendations = {
            item_id: predicted_scores[self.user_item_matrix.columns.get_loc(item_id)]
            for item_id in unrated_items.index
        }

        # Sort and return top-N
        return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def _item_based_recommend(self, user_id: int, top_n: int = 10):
        if user_id not in self.user_item_matrix.index:
            return []

        user_ratings = self.user_item_matrix.loc[user_id]
        scored_items = {}

        for item_id, rating in user_ratings.items():
            if rating == 0:
                continue
            item_idx = self.user_item_matrix.columns.get_loc(item_id)
            similarity_scores = self.similarity_matrix[item_idx]
            for i, score in enumerate(similarity_scores):
                target_item = self.user_item_matrix.columns[i]
                if user_ratings[target_item] == 0:
                    scored_items[target_item] = scored_items.get(target_item, 0) + rating * score

        # Sort and return top-N
        sorted_scores = sorted(scored_items.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_n]
