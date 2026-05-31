from typing import Dict, Iterable, List, Sequence


def rank_of_label(scores: Sequence[float], label_index: int) -> int:
    ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    return ranked_indices.index(label_index) + 1


def mean_reciprocal_rank(ranks: Iterable[int]) -> float:
    ranks = list(ranks)
    if not ranks:
        return 0.0
    return sum(1.0 / rank for rank in ranks) / len(ranks)


def recall_at_k(ranks: Iterable[int], k: int) -> float:
    ranks = list(ranks)
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank <= k) / len(ranks)


def compute_ranking_metrics(ranks: List[int]) -> Dict[str, float]:
    return {
        "mrr": mean_reciprocal_rank(ranks),
        "recall_at_1": recall_at_k(ranks, 1),
        "recall_at_2": recall_at_k(ranks, 2),
        "recall_at_5": recall_at_k(ranks, 5),
        "num_examples": float(len(ranks)),
    }

