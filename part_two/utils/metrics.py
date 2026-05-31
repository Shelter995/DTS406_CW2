from collections import Counter
from typing import Dict, Iterable, List, Sequence

from part_two.utils.text_utils import word_tokens


def rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    pred_tokens = word_tokens(prediction)
    ref_tokens = word_tokens(reference)
    rouge_1 = _rouge_n(pred_tokens, ref_tokens, 1)
    rouge_2 = _rouge_n(pred_tokens, ref_tokens, 2)
    rouge_l = _rouge_l(pred_tokens, ref_tokens)
    return {
        "rouge1_f1": rouge_1,
        "rouge2_f1": rouge_2,
        "rougeL_f1": rouge_l,
    }


def average_rouge(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {"rouge1_f1": 0.0, "rouge2_f1": 0.0, "rougeL_f1": 0.0}
    keys = ["rouge1_f1", "rouge2_f1", "rougeL_f1"]
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def corpus_bleu(predictions: Sequence[str], references: Sequence[str]) -> float:
    pred_tokens = [word_tokens(text) for text in predictions]
    ref_tokens = [word_tokens(text) for text in references]
    if not pred_tokens:
        return 0.0

    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu as nltk_corpus_bleu

        refs = [[tokens] for tokens in ref_tokens]
        return float(
            nltk_corpus_bleu(
                refs,
                pred_tokens,
                smoothing_function=SmoothingFunction().method1,
            )
        )
    except Exception:
        return _simple_corpus_bleu(pred_tokens, ref_tokens)


def _rouge_n(pred_tokens: Sequence[str], ref_tokens: Sequence[str], n: int) -> float:
    if len(pred_tokens) < n or len(ref_tokens) < n:
        return 0.0
    pred_ngrams = Counter(_ngrams(pred_tokens, n))
    ref_ngrams = Counter(_ngrams(ref_tokens, n))
    overlap = sum((pred_ngrams & ref_ngrams).values())
    return _f1(overlap, sum(pred_ngrams.values()), sum(ref_ngrams.values()))


def _rouge_l(pred_tokens: Sequence[str], ref_tokens: Sequence[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = _lcs_length(pred_tokens, ref_tokens)
    return _f1(overlap, len(pred_tokens), len(ref_tokens))


def _f1(overlap: int, pred_total: int, ref_total: int) -> float:
    if overlap <= 0 or pred_total <= 0 or ref_total <= 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    return 2 * precision * recall / (precision + recall)


def _ngrams(tokens: Sequence[str], n: int) -> List[tuple]:
    return [tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1)]


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for idx_b, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[idx_b - 1] + 1)
            else:
                current.append(max(previous[idx_b], current[-1]))
        previous = current
    return previous[-1]


def _simple_corpus_bleu(predictions: Sequence[Sequence[str]], references: Sequence[Sequence[str]]) -> float:
    matches = 0
    total = 0
    pred_len = 0
    ref_len = 0
    for pred, ref in zip(predictions, references):
        pred_counter = Counter(pred)
        ref_counter = Counter(ref)
        matches += sum((pred_counter & ref_counter).values())
        total += len(pred)
        pred_len += len(pred)
        ref_len += len(ref)
    if total == 0:
        return 0.0
    precision = matches / total
    brevity_penalty = 1.0 if pred_len > ref_len else pow(2.718281828, 1 - ref_len / max(pred_len, 1))
    return brevity_penalty * precision

