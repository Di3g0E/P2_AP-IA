import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from utils.config import TFIDF_PARAMS, SGD_PARAMS

class FinancialClassifier:
    """Resource-efficient classifier with incremental learning support."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
        self.clf = SGDClassifier(**SGD_PARAMS)
        self.classes_ = None

    def fit(self, X, y):
        X_vec = self.vectorizer.fit_transform(X)
        self.clf.fit(X_vec, y)
        self.classes_ = self.clf.classes_

    def partial_fit(self, X, y):
        X_vec = self.vectorizer.transform(X)
        self.clf.partial_fit(X_vec, y, classes=self.classes_)

    def predict(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.clf.predict(X_vec)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'vectorizer': self.vectorizer, 
            'clf': self.clf, 
            'classes': self.classes_
        }, path)

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        data = joblib.load(path)
        instance = cls()
        instance.vectorizer = data['vectorizer']
        instance.clf = data['clf']
        instance.classes_ = data['classes']
        return instance
