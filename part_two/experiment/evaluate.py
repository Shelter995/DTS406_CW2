from typing import Dict, List

from part_two.experiment.config import PartTwoConfig
from part_two.utils.io_utils import read_dicts_csv, write_dicts_csv, write_json
from part_two.utils.log_utils import log_message
from part_two.utils.metrics import average_rouge, corpus_bleu, rouge_scores


def evaluate_summaries(config: PartTwoConfig) -> None:
    rows = read_dicts_csv(config.generation_results_path)
    if not rows:
        raise ValueError(f"No generation results found in {config.generation_results_path}")

    rouge_rows: List[Dict[str, object]] = []
    baseline_rouge: List[Dict[str, float]] = []
    rag_rouge: List[Dict[str, float]] = []
    baseline_predictions: List[str] = []
    rag_predictions: List[str] = []
    references: List[str] = []

    for row in rows:
        reference = row["reference"]
        baseline_summary = row["baseline_summary"]
        rag_summary = row["rag_summary"]

        baseline_scores = rouge_scores(baseline_summary, reference)
        rag_scores = rouge_scores(rag_summary, reference)
        baseline_rouge.append(baseline_scores)
        rag_rouge.append(rag_scores)
        baseline_predictions.append(baseline_summary)
        rag_predictions.append(rag_summary)
        references.append(reference)

        rouge_rows.append(
            {
                "id": row["id"],
                "baseline_rouge1_f1": f"{baseline_scores['rouge1_f1']:.6f}",
                "baseline_rouge2_f1": f"{baseline_scores['rouge2_f1']:.6f}",
                "baseline_rougeL_f1": f"{baseline_scores['rougeL_f1']:.6f}",
                "rag_rouge1_f1": f"{rag_scores['rouge1_f1']:.6f}",
                "rag_rouge2_f1": f"{rag_scores['rouge2_f1']:.6f}",
                "rag_rougeL_f1": f"{rag_scores['rougeL_f1']:.6f}",
            }
        )

    baseline_avg = average_rouge(baseline_rouge)
    rag_avg = average_rouge(rag_rouge)
    metrics = {
        "num_examples": len(rows),
        "baseline": {
            **baseline_avg,
            "bleu": corpus_bleu(baseline_predictions, references),
        },
        "rag": {
            **rag_avg,
            "bleu": corpus_bleu(rag_predictions, references),
        },
    }

    write_dicts_csv(
        config.rouge_results_path,
        rouge_rows,
        [
            "id",
            "baseline_rouge1_f1",
            "baseline_rouge2_f1",
            "baseline_rougeL_f1",
            "rag_rouge1_f1",
            "rag_rouge2_f1",
            "rag_rougeL_f1",
        ],
    )
    write_json(config.final_metrics_path, metrics)
    _write_qualitative_examples(config, rows, rouge_rows)
    log_message(config.run_log_path, f"Wrote final metrics to {config.final_metrics_path}")


def _write_qualitative_examples(
    config: PartTwoConfig,
    rows: List[Dict[str, str]],
    rouge_rows: List[Dict[str, object]],
) -> None:
    selected = []
    for row, rouge_row in zip(rows, rouge_rows):
        baseline_l = float(rouge_row["baseline_rougeL_f1"])
        rag_l = float(rouge_row["rag_rougeL_f1"])
        selected.append((abs(rag_l - baseline_l), row))
    selected.sort(key=lambda item: item[0], reverse=True)

    examples = []
    for _, row in selected[: config.qualitative_example_count]:
        examples.append(
            {
                "id": row["id"],
                "reference": row["reference"],
                "baseline_summary": row["baseline_summary"],
                "rag_summary": row["rag_summary"],
                "retrieved_chunks": row["retrieved_chunks"],
            }
        )

    write_dicts_csv(
        config.qualitative_examples_path,
        examples,
        ["id", "reference", "baseline_summary", "rag_summary", "retrieved_chunks"],
    )

