import os
import logging

# --- PATHS ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "db_mod_descript_train.csv")
MODEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models", "LinearSVM_mod.joblib")

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
