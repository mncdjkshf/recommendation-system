# app/api/schemas.py

from dataclasses import dataclass
from typing import List, Union

@dataclass
class RecommendationRequest:
    user_id: int
    top_n: int = 10

@dataclass
class ItemSimilarityRequest:
    item_id: int
    top_n: int = 10

@dataclass
class Recommendation:
    item_id: int
    score: float
    title: Union[str, None] = None

@dataclass
class RecommendationResponse:
    user_id: int
    recommendations: List[Recommendation]

@dataclass
class SimilarItemsResponse:
    item_id: int
    similar_items: List[Recommendation]
