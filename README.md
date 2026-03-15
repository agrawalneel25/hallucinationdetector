# Hallucination Detection via TF-IDF Claim Verification

A lightweight, pure-NumPy approach to detecting hallucinated claims in generated text.

## The Problem

A simple but practical detection strategy is **retrieval grounding**: score each generated claim against a reference document and flag claims that are poorly supported.

## Approach

Given a reference document and a set of claims, we:

1. Represent the reference and each claim as TF-IDF vectors in a shared vocabulary.
2. Score each claim by its cosine similarity to the reference vector.
3. Flag claims below a similarity threshold as potential hallucinations.

This is a **lexical grounding** approach - it detects when a claim uses vocabulary that is absent or rare in the reference. 

However, it will miss hallucinations that are lexically consistent but semantically wrong.

## TF-IDF

For a term `t` in document `d` across a corpus of `N` documents:

```
TF(t, d)  = count(t in d) / total tokens in d
IDF(t)    = log((1 + N) / (1 + df(t))) + 1     # smooth IDF
TF-IDF    = TF * IDF
```

Each document vector is L2-normalised, so cosine similarity reduces to a dot product.

IDF is fitted on the reference document. Claims are projected into the same space using the pre-fitted IDF - they are strictly unseen at fit time.

## Structure

```
data/
  samples.py     labelled reference/claim pairs
src/
  preprocess.py  tokenisation, stopword removal, vocabulary
  tfidf.py       TF-IDF matrix construction in NumPy
  detector.py    scoring, thresholding, precision/recall/F1
main.py          end-to-end demo
```

## Running

```bash
python main.py
```

No dependencies beyond NumPy and the standard library.

## Limitations

- Lexical only - misses hallucinations that reuse reference vocabulary with wrong facts (e.g. swapping a date).
- Threshold is fixed; a real system would tune it on a validation set.
- Short claims against short references produce low absolute similarity scores regardless of groundedness. Better suited for sentence-level claims against paragraph-length references.

## Extensions

- Replace TF-IDF with dense embeddings
- (e.g. from a trained word2vec model - this is something I could combine with another application I made for the JetBrains internship!)
  for semantic rather than lexical similarity.
- Use sentence segmentation to automatically split generated output into claims before scoring.
- Add a calibration step to convert raw scores into calibrated probabilities.
