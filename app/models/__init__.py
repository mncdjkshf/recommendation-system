# app/models/__init__.py

def get_model(model_type: str):
    if model_type == "collaborative":
        from .collaborative import CollaborativeFiltering
        return CollaborativeFiltering()
    elif model_type == "content_based":
        from .content_based import ContentBasedFiltering
        return ContentBasedFiltering()
    elif model_type == "hybrid":
        from .hybrid import HybridRecommender
        return HybridRecommender()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
