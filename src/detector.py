"""
Core hallucination detection logic.

A claim is considered grounded if its TF-IDF vector is sufficiently similar
to the reference document vector. Cosine similarity is the scoring function —
after L2 normalisation in tfidf.py, this reduces to a dot product.

The threshold is the only hyperparameter. It should be tuned on labelled data
in a real system; here we treat 0.1 as a reasonable default for short claims
against a short reference paragraph.
"""

import numpy as np
from typing import List, Tuple
from src.preprocess import build_vocabulary
from src.tfidf import fit_transform, transform


def score_claims(
    reference: str,
    claims: List[str],
    threshold: float = 0.1,
) -> List[Tuple[str, float, bool]]:
    """
    Score each claim against the reference and return a verdict.

    Returns a list of (claim_text, similarity_score, is_flagged) tuples.
    is_flagged=True means the claim looks like a hallucination.
    """
    all_texts = [reference] + claims
    vocab = build_vocabulary(all_texts)

    # Fit IDF on the reference only — claims are unseen at fit time.
    ref_matrix, idf = fit_transform([reference], vocab)
    ref_vec = ref_matrix[0]  # shape: (vocab_size,)

    claim_matrix = transform(claims, vocab, idf)  # shape: (n_claims, vocab_size)

    # Cosine similarity: vectors are already L2-normalised, so dot product suffices.
    scores = claim_matrix @ ref_vec  # shape: (n_claims,)

    results = []
    for claim, score in zip(claims, scores):
        flagged = score < threshold
        results.append((claim.strip(), float(score), flagged))

    return results


def evaluate(
    results: List[Tuple[str, float, bool]],
    labels: List[bool],
) -> dict:
    """
    Compare detector output against ground-truth labels.
    labels[i]=True means the claim is grounded (not a hallucination).
    """
    tp = fp = tn = fn = 0
    for (_, _, flagged), grounded in zip(results, labels):
        predicted_hallucination = flagged
        actual_hallucination = not grounded
        if predicted_hallucination and actual_hallucination:
            tp += 1
        elif predicted_hallucination and not actual_hallucination:
            fp += 1
        elif not predicted_hallucination and not actual_hallucination:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}
