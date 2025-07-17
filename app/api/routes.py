# app/api/routes.py

from flask import Blueprint, request, jsonify
from app.services.recommender_service import RecommenderService

# Define blueprint for API routes
api_blueprint = Blueprint('api', __name__)

# Initialize the recommendation service
recommender = RecommenderService()

@api_blueprint.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running!"}), 200


@api_blueprint.route('/recommend/user/<int:user_id>', methods=['GET'])
def recommend_for_user(user_id):
    """
    Recommend items to a user using collaborative filtering.
    Example: GET /api/recommend/user/42
    """
    try:
        top_n = int(request.args.get("top_n", 10))
        recommendations = recommender.recommend_for_user(user_id, top_n=top_n)
        return jsonify({"user_id": user_id, "recommendations": recommendations}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_blueprint.route('/recommend/item/<int:item_id>', methods=['GET'])
def recommend_similar_items(item_id):
    """
    Recommend similar items using content-based filtering.
    Example: GET /api/recommend/item/15
    """
    try:
        top_n = int(request.args.get("top_n", 10))
        recommendations = recommender.recommend_similar_items(item_id, top_n=top_n)
        return jsonify({"item_id": item_id, "recommendations": recommendations}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
