# app/data/__init__.py

"""
This module handles access to datasets used in the recommendation system.
You can use this file to expose common data loaders or constants.
"""

from .loaders import load_ratings_data, load_items_data

__all__ = [
    "load_ratings_data",
    "load_items_data"
]
