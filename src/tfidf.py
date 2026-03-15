# TF-IDF from scratch.
#
# TF(t, d)  = count(t in d) / len(d)
# IDF(t)    = log((1 + N) / (1 + df(t))) + 1   <- sklearn-style smoothing
# TF-IDF    = TF * IDF, then L2-normalise rows so cosine sim = dot product

import numpy as np
from typing import List
from src.preprocess import tokenise


def _term_freq(tokens: List[str], vocab: List[str]) -> np.ndarray:
    index = {w: i for i, w in enumerate(vocab)}
    tf = np.zeros(len(vocab), dtype=np.float64)
    for t in tokens:
        if t in index:
            tf[index[t]] += 1
    if tf.sum() > 0:
        tf /= tf.sum()
    return tf


def fit_transform(documents: List[str], vocab: List[str]):
    """Build TF-IDF matrix and return it alongside the fitted IDF vector."""
    N = len(documents)
    tf_matrix = np.zeros((N, len(vocab)), dtype=np.float64)
    for i, doc in enumerate(documents):
        tf_matrix[i] = _term_freq(tokenise(doc), vocab)

    df = np.sum(tf_matrix > 0, axis=0).astype(np.float64)
    idf = np.log((1 + N) / (1 + df)) + 1
    tfidf = tf_matrix * idf

    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return tfidf / norms, idf


def transform(documents: List[str], vocab: List[str], idf: np.ndarray) -> np.ndarray:
    """Project new documents into the space defined by a pre-fitted IDF."""
    N = len(documents)
    tf_matrix = np.zeros((N, len(vocab)), dtype=np.float64)
    for i, doc in enumerate(documents):
        tf_matrix[i] = _term_freq(tokenise(doc), vocab)

    tfidf = tf_matrix * idf
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return tfidf / norms
