from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt

from part_two.experiment.config import get_config
from part_two.utils.io_utils import ensure_dir, read_dicts_csv, read_json
from part_two.utils.text_utils import word_tokens


def main() -> None:
    config = get_config()
    ensure_dir(config.figures_dir)

    metrics = read_json(config.final_metrics_path)
    generation_rows = read_dicts_csv(config.generation_results_path)
    stats = read_json(config.data_statistics_path)

    plot_rouge_comparison(metrics, config.figures_dir / "rouge_comparison.png")
    plot_bleu_comparison(metrics, config.figures_dir / "bleu_comparison.png")
    plot_length_statistics(generation_rows, config.figures_dir / "length_statistics.png")
    plot_compression_ratio(generation_rows, config.figures_dir / "compression_ratio.png")
    plot_data_statistics(stats, config.figures_dir / "data_statistics.png")

    print(f"Saved Part Two figures to {config.figures_dir}")


def plot_rouge_comparison(metrics, output_path: Path) -> None:
    labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    baseline = [
        float(metrics["baseline"]["rouge1_f1"]),
        float(metrics["baseline"]["rouge2_f1"]),
        float(metrics["baseline"]["rougeL_f1"]),
    ]
    rag = [
        float(metrics["rag"]["rouge1_f1"]),
        float(metrics["rag"]["rouge2_f1"]),
        float(metrics["rag"]["rougeL_f1"]),
    ]

    x = range(len(labels))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar([idx - width / 2 for idx in x], baseline, width=width, label="Baseline")
    plt.bar([idx + width / 2 for idx in x], rag, width=width, label="RAG")
    plt.xticks(list(x), labels)
    plt.ylabel("Average F1")
    plt.title("ROUGE Comparison")
    plt.ylim(0, max(baseline + rag + [0.1]) * 1.2)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_bleu_comparison(metrics, output_path: Path) -> None:
    labels = ["Baseline", "RAG"]
    values = [float(metrics["baseline"]["bleu"]), float(metrics["rag"]["bleu"])]
    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, values, color=["#4c78a8", "#59a14f"])
    plt.ylabel("Corpus BLEU")
    plt.title("BLEU Comparison")
    plt.ylim(0, max(values + [0.1]) * 1.25)
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values + [0.1]) * 0.03,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_length_statistics(rows, output_path: Path) -> None:
    reference_lengths = [len(word_tokens(row["reference"])) for row in rows]
    baseline_lengths = [len(word_tokens(row["baseline_summary"])) for row in rows]
    rag_lengths = [len(word_tokens(row["rag_summary"])) for row in rows]

    labels = ["Reference", "Baseline", "RAG"]
    values = [
        _mean(reference_lengths),
        _mean(baseline_lengths),
        _mean(rag_lengths),
    ]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values, color=["#999999", "#4c78a8", "#59a14f"])
    plt.ylabel("Average Word Count")
    plt.title("Summary Length Comparison")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_compression_ratio(rows, output_path: Path) -> None:
    baseline_ratios = []
    rag_ratios = []
    for row in rows:
        article_len = len(word_tokens(row["article"]))
        if article_len == 0:
            continue
        baseline_ratios.append(len(word_tokens(row["baseline_summary"])) / article_len)
        rag_ratios.append(len(word_tokens(row["rag_summary"])) / article_len)

    plt.figure(figsize=(8, 5))
    plt.hist(baseline_ratios, bins=30, alpha=0.65, label="Baseline")
    plt.hist(rag_ratios, bins=30, alpha=0.65, label="RAG")
    plt.xlabel("Generated Summary Length / Article Length")
    plt.ylabel("Number of Examples")
    plt.title("Compression Ratio Distribution")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_data_statistics(stats, output_path: Path) -> None:
    labels = ["Article", "Reference"]
    values = [
        float(stats.get("avg_article_words", 0.0)),
        float(stats.get("avg_reference_words", 0.0)),
    ]
    plt.figure(figsize=(6, 5))
    plt.bar(labels, values, color=["#4c78a8", "#f28e2b"])
    plt.ylabel("Average Word Count")
    plt.title("CNN/DailyMail Length Statistics")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _mean(values) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
