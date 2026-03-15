import re
from typing import List

# not using nltk — keeping it self-contained
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "has", "had", "have", "it", "its", "this", "that", "as", "not", "no",
    "who", "which", "she", "he", "they", "her", "his", "their", "also",
}


def tokenise(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def build_vocabulary(documents: List[str]) -> List[str]:
    vocab = set()
    for doc in documents:
        vocab.update(tokenise(doc))
    return sorted(vocab)
