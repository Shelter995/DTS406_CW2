from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import csv
import json
from typing import Dict, List

import matplotlib.pyplot as plt

from part_one.experiment.config import get_config
from part_one.utils.io_utils import ensure_dir


def main() -> None:
    config = get_config()
    figures_dir = config.results_dir / "figures"
    ensure_dir(figures_dir)

    validation_rows = _read_csv(config.validation_metrics_path)
    test_metrics = _read_json(config.test_metrics_path)
    prediction_rows = _read_csv(config.test_predictions_path)
    data_statistics = _read_json(config.data_statistics_path)

    plot_validation_metrics(validation_rows, figures_dir / "validation_metrics.png")
    plot_training_loss(validation_rows, figures_dir / "training_loss.png")
    plot_correct_rank_distribution(
        prediction_rows,
        figures_dir / "correct_rank_distribution.png",
    )
    plot_recall_at_k(test_metrics, figures_dir / "recall_at_k.png")
    plot_score_gap_distribution(
        prediction_rows,
        figures_dir / "score_gap_distribution.png",
    )
    plot_data_statistics(data_statistics, figures_dir / "data_statistics.png")

    print(f"Saved figures to {figures_dir}")


def plot_validation_metrics(rows: List[Dict[str, str]], output_path: Path) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    mrr = [_to_float(row["mrr"]) for row in rows]
    recall_1 = [_to_float(row["recall_at_1"]) for row in rows]
    recall_2 = [_to_float(row["recall_at_2"]) for row in rows]
    recall_5 = [_to_float(row["recall_at_5"]) for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, mrr, marker="o", label="MRR")
    plt.plot(epochs, recall_1, marker="o", label="Recall@1")
    plt.plot(epochs, recall_2, marker="o", label="Recall@2")
    plt.plot(epochs, recall_5, marker="o", label="Recall@5")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Ranking Metrics by Epoch")
    plt.xticks(epochs)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_training_loss(rows: List[Dict[str, str]], output_path: Path) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    losses = [_to_float(row["train_loss"]) for row in rows]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, losses, marker="o", color="#2f5597")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss by Epoch")
    plt.xticks(epochs)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_correct_rank_distribution(rows: List[Dict[str, str]], output_path: Path) -> None:
    rank_counts = {rank: 0 for rank in range(1, 11)}
    for row in rows:
        rank = int(row["correct_rank"])
        rank_counts[rank] = rank_counts.get(rank, 0) + 1

    ranks = list(range(1, 11))
    counts = [rank_counts.get(rank, 0) for rank in ranks]

    plt.figure(figsize=(8, 5))
    plt.bar(ranks, counts, color="#4c78a8")
    plt.xlabel("Correct Response Rank")
    plt.ylabel("Number of Test Examples")
    plt.title("Correct Rank Distribution on Test Set")
    plt.xticks(ranks)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_recall_at_k(metrics: Dict[str, object], output_path: Path) -> None:
    labels = ["Recall@1", "Recall@2", "Recall@5"]
    values = [
        float(metrics["recall_at_1"]),
        float(metrics["recall_at_2"]),
        float(metrics["recall_at_5"]),
    ]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values, color=["#4c78a8", "#59a14f", "#f28e2b"])
    plt.ylabel("Recall")
    plt.title("Test Recall@k")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_score_gap_distribution(rows: List[Dict[str, str]], output_path: Path) -> None:
    gaps = []
    for row in rows:
        label = int(row["label"])
        correct_score = _to_float(row[f"score_{label}"])
        negative_scores = [
            _to_float(row[f"score_{idx}"]) for idx in range(10) if idx != label
        ]
        best_negative = max(negative_scores)
        gaps.append(correct_score - best_negative)

    plt.figure(figsize=(8, 5))
    plt.hist(gaps, bins=40, color="#8064a2", edgecolor="white")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Correct Score - Best Negative Score")
    plt.ylabel("Number of Test Examples")
    plt.title("Score Gap Distribution")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_data_statistics(stats: Dict[str, object], output_path: Path) -> None:
    train_stats = stats.get("train_100k", {})
    valid_stats = stats.get("valid_5k", {})
    test_stats = stats.get("test_5k", {})

    labels = [
        "Train Context",
        "Train Response",
        "Valid Context",
        "Valid Candidate",
        "Test Context",
        "Test Candidate",
    ]
    values = [
        float(train_stats.get("avg_context_tokens", 0)),
        float(train_stats.get("avg_response_tokens", 0)),
        float(valid_stats.get("avg_context_tokens", 0)),
        float(valid_stats.get("avg_candidate_tokens", 0)),
        float(test_stats.get("avg_context_tokens", 0)),
        float(test_stats.get("avg_candidate_tokens", 0)),
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color="#5b9bd5")
    plt.ylabel("Average Word Tokens")
    plt.title("Average Text Lengths After Preprocessing")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required result file does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Required result file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_float(value: object) -> float:
    return float(str(value).strip())


if __name__ == "__main__":
    main()

