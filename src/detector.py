import numpy as np
from typing import List, Tuple
from src.preprocess import build_vocabulary
from src.tfidf import fit_transform, transform


def score_claims(
    reference: str,
    claims: List[str],
    threshold: float = 0.15,
) -> List[Tuple[str, float, bool]]:
    """
    Score each claim by its TF-IDF cosine similarity to the reference.
    Claims below the threshold are flagged as likely hallucinations.
    IDF is fitted on the reference only — claims are unseen at fit time.
    """
    vocab = build_vocabulary([reference] + claims)

    ref_matrix, idf = fit_transform([reference], vocab)
    ref_vec = ref_matrix[0]

    claim_matrix = transform(claims, vocab, idf)
    scores = claim_matrix @ ref_vec  # dot product == cosine sim after L2 norm

    return [(c.strip(), float(s), s < threshold) for c, s in zip(claims, scores)]


def evaluate(results: List[Tuple[str, float, bool]], labels: List[bool]) -> dict:
    tp = fp = tn = fn = 0
    for (_, _, flagged), grounded in zip(results, labels):
        hallucination = not grounded
        if flagged and hallucination:
            tp += 1
        elif flagged and not hallucination:
            fp += 1
        elif not flagged and not hallucination:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}
