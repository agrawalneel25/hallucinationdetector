"""
Hallucination detection demo.

Scores each claim against a reference document using TF-IDF cosine similarity.
Claims that are poorly supported by the reference are flagged as hallucinations.
"""

from data.samples import EXAMPLES
from src.detector import score_claims, evaluate

THRESHOLD = 0.15


def run(example: dict) -> None:
    reference = example["reference"]
    claim_texts = [c for c, _ in example["claims"]]
    labels      = [grounded for _, grounded in example["claims"]]

    results = score_claims(reference, claim_texts, threshold=THRESHOLD)

    print("Reference:")
    print(" ", reference.strip()[:120], "...")
    print()
    print(f"{'Claim':<55}  {'Score':>6}  {'Verdict'}")
    print("-" * 75)
    for claim, score, flagged in results:
        verdict = "HALLUCINATION" if flagged else "grounded"
        print(f"  {claim:<53}  {score:.3f}  {verdict}")

    print()
    metrics = evaluate(results, labels)
    print(f"  Precision: {metrics['precision']:.2f}  "
          f"Recall: {metrics['recall']:.2f}  "
          f"F1: {metrics['f1']:.2f}")
    print(f"  (TP={metrics['tp']} FP={metrics['fp']} "
          f"TN={metrics['tn']} FN={metrics['fn']})")
    print()


if __name__ == "__main__":
    for i, example in enumerate(EXAMPLES, 1):
        print(f"{'=' * 75}")
        print(f"  Example {i}")
        print(f"{'=' * 75}")
        run(example)
