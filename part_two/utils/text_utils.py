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
    if text is None:
        return ""
    text = str(text).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sentence_split(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    try:
        import nltk

        return [sentence.strip() for sentence in nltk.sent_tokenize(text) if sentence.strip()]
    except Exception:
        pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
        return [piece.strip() for piece in pieces if piece.strip()]


def word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_']+", text.lower())


def remove_stopwords(tokens: Iterable[str]) -> List[str]:
    return [token for token in tokens if token not in STOPWORDS]

