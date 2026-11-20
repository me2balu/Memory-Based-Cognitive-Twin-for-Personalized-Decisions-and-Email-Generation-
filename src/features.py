"""
features.py
------------
Text feature extraction for the ML decision-prediction task.

Two feature representations are provided:
  1. TF-IDF (sparse)   — classical baseline features
  2. Sentence embeddings (dense) — semantic features from the memory module

LEAKAGE PREVENTION:
  - TF-IDF vectorizer is fitted ONLY on training texts.
  - Sentence-embedding model is pre-trained; no leakage possible.
  - No statistics from val/test are used anywhere in this module.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from typing import Tuple
from src.memory import embed_texts   # reuses the singleton model & cache


# -- Label encoding ------------------------------------------------------------
DECISION_CLASSES = ["ACCEPT", "DEFER", "MAYBE", "REJECT", "REQUEST_MORE_INFORMATION"]


def make_label_encoder() -> LabelEncoder:
    le = LabelEncoder()
    le.fit(DECISION_CLASSES)
    return le


# -- TF-IDF features -----------------------------------------------------------

def fit_tfidf(
    train_texts: list[str],
    max_features: int = 3000,
    ngram_range: tuple = (1, 2),
) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectorizer on training texts only.
    max_features=3000 keeps computation fast while capturing most vocabulary.
    """
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,    # log(1 + tf) dampens very frequent terms
        min_df=2,             # ignore terms seen in only one document
    )
    vec.fit(train_texts)
    return vec


def tfidf_features(
    vec: TfidfVectorizer,
    texts: list[str],
) -> np.ndarray:
    """Transform texts to dense TF-IDF feature matrix (n, max_features)."""
    return vec.transform(texts).toarray().astype(np.float32)


# -- Sentence embedding features -----------------------------------------------

def embedding_features(texts: list[str]) -> np.ndarray:
    """
    Return (n, 384) sentence embedding matrix.
    Internally uses embed_texts() which caches results to disk.
    """
    return embed_texts(texts)


# -- Combined feature set ------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    tfidf_vec: TfidfVectorizer | None = None,
    use_embeddings: bool = True,
    use_tfidf: bool = True,
) -> Tuple[np.ndarray, TfidfVectorizer | None]:
    """
    Build the feature matrix X for the given dataframe.

    Parameters:
        df             : dataframe (train / val / test slice)
        tfidf_vec      : fitted vectorizer; if None and use_tfidf=True, raises.
        use_embeddings : include sentence embeddings (recommended for CognitiveTwin)
        use_tfidf      : include TF-IDF features (used by baseline)

    Returns:
        (X, tfidf_vec) where X has shape (n, d).
    """
    texts = df["incoming_email"].tolist()
    parts = []

    if use_tfidf:
        if tfidf_vec is None:
            raise ValueError("Pass a fitted TfidfVectorizer for use_tfidf=True.")
        parts.append(tfidf_features(tfidf_vec, texts))

    if use_embeddings:
        parts.append(embedding_features(texts))

    X = np.hstack(parts) if len(parts) > 1 else parts[0]
    return X, tfidf_vec
