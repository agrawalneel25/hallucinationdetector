from data.samples import EXAMPLES
from src.detector import score_claims, evaluate

THRESHOLD = 0.15


def run(example: dict) -> None:
    reference = example["reference"]
    claims    = [c for c, _ in example["claims"]]
    labels    = [grounded for _, grounded in example["claims"]]

    results = score_claims(reference, claims, threshold=THRESHOLD)

    print("Reference:")
    print(" ", reference.strip()[:120], "...")
    print()
    print(f"{'Claim':<55}  {'Score':>6}  Verdict")
    print("-" * 75)
    for claim, score, flagged in results:
        verdict = "HALLUCINATION" if flagged else "grounded"
        print(f"  {claim:<53}  {score:.3f}  {verdict}")

    metrics = evaluate(results, labels)
    print(f"\n  Precision: {metrics['precision']:.2f}  "
          f"Recall: {metrics['recall']:.2f}  "
          f"F1: {metrics['f1']:.2f}  "
          f"(TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']})\n")


if __name__ == "__main__":
    for i, example in enumerate(EXAMPLES, 1):
        print("=" * 75)
        print(f"  Example {i}")
        print("=" * 75)
        run(example)
