# app/services/recommender_service.py

from app.models.collaborative import CollaborativeFiltering
from app.models.content_based import ContentBasedFiltering
from app.models.hybrid import HybridRecommender
import pandas as pd

class RecommenderService:
    def __init__(self):
        """
        Initializes and loads models + data.
        In production, this can load pre-trained models from disk.
        """
        # Example datasets (replace with real loading logic)
        self.ratings_df = self._load_ratings_data()
        self.items_df = self._load_items_data()

        # Initialize model instances
        self.cf_model = CollaborativeFiltering(self.ratings_df, similarity_type="user")
        self.cb_model = ContentBasedFiltering(self.items_df, text_column="description")
        self.hybrid_model = HybridRecommender(self.cf_model, self.cb_model, alpha=0.7)

    def recommend_for_user(self, user_id: int, top_n: int = 10) -> list:
        """
        Returns top-N hybrid recommendations for a user.
        """
        return self.hybrid_model.recommend_for_user(user_id, top_n)

    def recommend_similar_items(self, item_id: int, top_n: int = 10) -> list:
        """
        Returns top-N content-based similar items.
        """
        return self.cb_model.recommend_similar_items(item_id, top_n)

    # 🔽 Dummy dataset loading functions below (replace in production)

    def _load_ratings_data(self) -> pd.DataFrame:
        """
        Dummy ratings DataFrame with [user_id, item_id, rating]
        Replace with real data loading logic (e.g., from database, CSV, etc.)
        """
        data = {
            "user_id": [1, 1, 2, 2, 3],
            "item_id": [101, 102, 101, 103, 104],
            "rating": [5, 3, 4, 2, 5]
        }
        return pd.DataFrame(data)

    def _load_items_data(self) -> pd.DataFrame:
        """
        Dummy items DataFrame with [item_id, description]
        Replace with real metadata (e.g., title, genre, description, etc.)
        """
        data = {
            "item_id": [101, 102, 103, 104],
            "description": [
                "Action and adventure movie",
                "Romantic love story in Paris",
                "Sci-fi with space exploration",
                "More action and thrilling story"
            ]
        }
        return pd.DataFrame(data)
