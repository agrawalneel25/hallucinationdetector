"""
Basic text preprocessing: lowercase, tokenise, remove stopwords.
Kept minimal — the goal is clean tokens, not perfect NLP.
"""

import re
from typing import List

# Common words that carry little factual signal.
# Keeping this list short and explicit rather than importing nltk.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "has", "had", "have", "it", "its", "this", "that", "as", "not", "no",
    "who", "which", "she", "he", "they", "her", "his", "their", "also",
}


def tokenise(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace, drop stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def build_vocabulary(documents: List[str]) -> List[str]:
    """Return a sorted list of unique tokens across all documents."""
    vocab = set()
    for doc in documents:
        vocab.update(tokenise(doc))
    return sorted(vocab)
