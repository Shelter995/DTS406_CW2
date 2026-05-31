import re
from typing import Iterable, List


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "you",
    "your",
}


def normalize_text(text: object) -> str:
    """Light normalization while preserving Ubuntu dialogue markers."""
    if text is None:
        return ""
    text = str(text).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_']+", text.lower())


def remove_stopwords(tokens: Iterable[str]) -> List[str]:
    return [token for token in tokens if token not in STOPWORDS]

