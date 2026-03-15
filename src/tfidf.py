"""
TF-IDF vectoriser implemented from scratch in NumPy.

TF(t, d)  = count of t in d / total tokens in d
IDF(t)    = log((1 + N) / (1 + df(t))) + 1   (sklearn-style smooth IDF)
TF-IDF    = TF * IDF

The smooth IDF prevents division by zero and keeps scores for very common
terms from collapsing to zero entirely.
"""

import numpy as np
from typing import List
from src.preprocess import tokenise


def _term_freq(tokens: List[str], vocab: List[str]) -> np.ndarray:
    """Raw term frequency vector for a single document."""
    index = {w: i for i, w in enumerate(vocab)}
    tf = np.zeros(len(vocab), dtype=np.float64)
    for t in tokens:
        if t in index:
            tf[index[t]] += 1
    if tf.sum() > 0:
        tf /= tf.sum()
    return tf


def fit_transform(documents: List[str], vocab: List[str]) -> np.ndarray:
    """
    Build a TF-IDF matrix of shape (n_docs, vocab_size).
    Each row is the TF-IDF vector for that document.
    """
    N = len(documents)
    tf_matrix = np.zeros((N, len(vocab)), dtype=np.float64)

    for i, doc in enumerate(documents):
        tf_matrix[i] = _term_freq(tokenise(doc), vocab)

    # document frequency: how many docs contain each term
    df = np.sum(tf_matrix > 0, axis=0).astype(np.float64)
    idf = np.log((1 + N) / (1 + df)) + 1

    tfidf = tf_matrix * idf

    # L2-normalise each row so cosine similarity is just a dot product
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return tfidf / norms, idf


def transform(documents: List[str], vocab: List[str], idf: np.ndarray) -> np.ndarray:
    """
    Apply a pre-fitted IDF vector to new documents.
    Used to project claims into the same TF-IDF space as the reference.
    """
    N = len(documents)
    tf_matrix = np.zeros((N, len(vocab)), dtype=np.float64)

    for i, doc in enumerate(documents):
        tf_matrix[i] = _term_freq(tokenise(doc), vocab)

    tfidf = tf_matrix * idf

    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return tfidf / norms
