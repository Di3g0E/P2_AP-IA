import os
import logging

# --- PATHS ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "db_orig.csv")
MODEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models", "production_model.joblib")

# --- MODEL PARAMETERS ---
TFIDF_PARAMS = {
    'analyzer': 'char_wb',
    'ngram_range': (2, 5),
    'max_features': 10000
}

SGD_PARAMS = {
    'loss': 'log_loss',
    'penalty': 'l2',
    'alpha': 0.0001,
    'random_state': 42
}
