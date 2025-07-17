# app/models/hybrid.py

from typing import List, Tuple
from app.models.collaborative import CollaborativeFiltering
from app.models.content_based import ContentBasedFiltering

class HybridRecommender:
    def __init__(self, cf_model: CollaborativeFiltering, cb_model: ContentBasedFiltering, alpha: float = 0.5):
        """
        :param cf_model: Trained collaborative filtering model
        :param cb_model: Trained content-based filtering model
        :param alpha: Weight for blending (0.0 = only content-based, 1.0 = only collaborative)
        """
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.alpha = alpha

    def recommend_for_user(self, user_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Combines CF and CBF recommendations for a user.
        Returns top-N items with hybrid scores.
        """
        cf_results = dict(self.cf_model.recommend_for_user(user_id, top_n=100))
        cb_results = {}

        # Extract user's rated items
        rated_items = set(self.cf_model.user_item_matrix.loc[user_id][
            self.cf_model.user_item_matrix.loc[user_id] > 0].index
        ) if user_id in self.cf_model.user_item_matrix.index else set()

        # Build CB recommendations by aggregating similar items from rated ones
        for item_id in rated_items:
            similar_items = self.cb_model.recommend_similar_items(item_id, top_n=20)
            for sim_item_id, sim_score in similar_items:
                if sim_item_id not in rated_items:
                    cb_results[sim_item_id] = cb_results.get(sim_item_id, 0) + sim_score

        # Normalize CB scores
        if cb_results:
            max_cb = max(cb_results.values())
            cb_results = {k: v / max_cb for k, v in cb_results.items()}

        # Normalize CF scores
        if cf_results:
            max_cf = max(cf_results.values())
            cf_results = {k: v / max_cf for k, v in cf_results.items()}

        # Merge both scores
        all_items = set(cf_results.keys()).union(cb_results.keys())
        hybrid_scores = {
            item_id: self.alpha * cf_results.get(item_id, 0) +
                      (1 - self.alpha) * cb_results.get(item_id, 0)
            for item_id in all_items
        }

        # Sort and return top-N
        return sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
