from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from part_one.experiment.config import PartOneConfig
from part_one.experiment.dataset import RankingDataset
from part_one.utils.io_utils import write_json
from part_one.utils.metrics import compute_ranking_metrics, rank_of_label
from part_one.utils.predict import save_error_examples, save_predictions


def evaluate_checkpoint(
    config: PartOneConfig,
    checkpoint_dir: Path,
    data_path: Path,
    save_outputs: bool = False,
) -> Dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    metrics, prediction_rows = evaluate_model(model, tokenizer, config, data_path, device)

    if save_outputs:
        write_json(config.test_metrics_path, metrics)
        save_predictions(config.test_predictions_path, prediction_rows)
        save_error_examples(
            config.error_examples_path,
            prediction_rows,
            limit=config.error_example_limit,
        )

    return metrics


def evaluate_model(
    model,
    tokenizer,
    config: PartOneConfig,
    data_path: Path,
    device: torch.device,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    dataset = RankingDataset(data_path, config.num_candidates)
    records = dataset.records()
    scores_by_example = _score_records(model, tokenizer, records, config, device)

    ranks: List[int] = []
    prediction_rows: List[Dict[str, object]] = []

    for record, scores in zip(records, scores_by_example):
        label = int(record["label"])
        correct_rank = rank_of_label(scores, label)
        ranks.append(correct_rank)
        predicted_index = max(range(len(scores)), key=lambda idx: scores[idx])

        row: Dict[str, object] = {
            "sample_id": record["sample_id"],
            "context": record["context"],
            "label": label,
            "predicted_index": predicted_index,
            "correct_rank": correct_rank,
        }
        for idx, candidate in enumerate(record["candidates"]):
            row[f"candidate_{idx}"] = candidate
        for idx, score in enumerate(scores):
            row[f"score_{idx}"] = f"{score:.8f}"
        prediction_rows.append(row)

    return compute_ranking_metrics(ranks), prediction_rows


def _score_records(
    model,
    tokenizer,
    records: List[Dict[str, object]],
    config: PartOneConfig,
    device: torch.device,
) -> List[List[float]]:
    model.eval()
    pairs = []
    for sample_idx, record in enumerate(records):
        for candidate_idx, candidate in enumerate(record["candidates"]):
            pairs.append((sample_idx, candidate_idx, record["context"], candidate))

    scores_by_example = [
        [0.0 for _ in range(config.num_candidates)] for _ in range(len(records))
    ]

    autocast_context = _autocast_context(config, device)
    with torch.no_grad():
        for start in range(0, len(pairs), config.eval_batch_size):
            batch = pairs[start : start + config.eval_batch_size]
            contexts = [item[2] for item in batch]
            candidates = [item[3] for item in batch]
            encoded = tokenizer(
                contexts,
                candidates,
                max_length=config.max_length,
                truncation="longest_first",
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with autocast_context:
                logits = model(**encoded).logits
            positive_scores = logits[:, 1].detach().float().cpu().tolist()

            for (sample_idx, candidate_idx, _, _), score in zip(batch, positive_scores):
                scores_by_example[sample_idx][candidate_idx] = float(score)

    return scores_by_example


def _autocast_context(config: PartOneConfig, device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if config.prefer_bf16 and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if config.allow_fp16_fallback:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()

