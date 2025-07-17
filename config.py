# app/config.py

import os

# -----------------------------
# BASE SETTINGS
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# DATA FILES
# -----------------------------

RATINGS_FILE = os.path.join(DATA_DIR, 'ratings.csv')
ITEMS_FILE = os.path.join(DATA_DIR, 'items.csv')

# -----------------------------
# COLLABORATIVE FILTERING
# -----------------------------

COLLAB_MODEL_FILE = os.path.join(MODEL_DIR, 'collaborative_model.pkl')
NUM_FACTORS = 50              # Latent features for MF models
NUM_EPOCHS = 20
LEARNING_RATE = 0.01

# -----------------------------
# CONTENT-BASED FILTERING
# -----------------------------

TFIDF_MAX_FEATURES = 1000
TFIDF_MATRIX_FILE = os.path.join(MODEL_DIR, 'tfidf_matrix.npz')
TFIDF_VECTORIZER_FILE = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')

# -----------------------------
# HYBRID MODEL
# -----------------------------

HYBRID_MODEL_FILE = os.path.join(MODEL_DIR, 'hybrid_model.pkl')

# -----------------------------
# API SETTINGS
# -----------------------------

API_HOST = "0.0.0.0"
API_PORT = 8000
DEBUG_MODE = True
