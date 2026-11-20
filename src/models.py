"""
models.py
----------
ML models for decision prediction.

Two models are implemented:
  1. Baseline  — Logistic Regression on TF-IDF features.
  2. CognitiveTwin ML — Random Forest on TF-IDF + sentence embeddings.

Both are standard scikit-learn estimators.  The CognitiveTwin model uses
richer features (semantic embeddings) to better capture decision patterns.

Note: The LLM is NOT used for prediction — this is a genuine ML experiment.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Dict, Any

RANDOM_STATE = 42


# -- Baseline: Logistic Regression on TF-IDF ----------------------------------

def build_baseline() -> LogisticRegression:
    """
    Simple logistic regression baseline.
    Regularisation (C=1.0) guards against overfitting on small data.
    max_iter=1000 ensures convergence on multi-class problems.
    """
    return LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_STATE,
        multi_class="multinomial",
        solver="lbfgs",
    )


# -- CognitiveTwin ML: Random Forest on TF-IDF + embeddings -------------------

def build_cognitive_twin() -> RandomForestClassifier:
    """
    Random Forest on the combined TF-IDF + sentence-embedding feature space.

    Why Random Forest?
      - Handles mixed feature scales without normalisation.
      - Naturally multiclass.
      - Feature importance is interpretable.
      - Robust to the relatively small dataset size (300 emails).

    n_estimators=200 for stability; max_depth=None lets trees grow fully.
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# -- Training wrapper ----------------------------------------------------------

def train_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> BaseEstimator:
    """Fit and return the model."""
    model.fit(X_train, y_train)
    return model


def predict(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Return predicted labels."""
    return model.predict(X)


def predict_proba(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Return class probability matrix (n, n_classes)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    raise AttributeError(f"{type(model).__name__} does not support predict_proba.")
