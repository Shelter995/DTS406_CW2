import csv
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from part_one.experiment.config import PartOneConfig
from part_one.utils.io_utils import ensure_dir, write_dicts_csv, write_json
from part_one.utils.log_utils import log_message
from part_one.utils.text_utils import normalize_text, remove_stopwords, word_tokens


def prepare_part_one_data(config: PartOneConfig) -> None:
    """Create fixed train/dev/valid/test files from raw Ubuntu v2 CSV files."""
    log_message(config.run_log_path, "Preparing Part One data files.")
    for path in [config.raw_dir, config.processed_dir, config.results_dir]:
        ensure_dir(path)

    train_records = _reservoir_sample(
        _iter_raw_train(config.raw_train_path),
        config.train_sample_size,
        config.seed,
    )
    train_dev_records = _reservoir_sample(
        _iter_raw_train(config.raw_train_path),
        config.train_dev_size,
        config.seed + 1,
    )

    if not train_records:
        raise ValueError(f"No train records were read from {config.raw_train_path}")

    _write_train_records(config.train_path, train_records)
    _write_train_records(config.train_dev_path, train_dev_records)
    log_message(
        config.run_log_path,
        f"Wrote {len(train_records)} training rows to {config.train_path}",
    )
    log_message(
        config.run_log_path,
        f"Wrote {len(train_dev_records)} dev training rows to {config.train_dev_path}",
    )

    _prepare_ranking_split(
        config=config,
        raw_path=config.raw_valid_path,
        output_path=config.valid_path,
        dev_output_path=config.valid_dev_path,
        main_size=config.valid_sample_size,
        dev_size=config.valid_dev_size,
        seed=config.seed,
        split_name="valid",
        num_candidates=config.num_candidates,
    )
    _prepare_ranking_split(
        config=config,
        raw_path=config.raw_test_path,
        output_path=config.test_path,
        dev_output_path=config.test_dev_path,
        main_size=config.test_sample_size,
        dev_size=config.test_dev_size,
        seed=config.seed,
        split_name="test",
        num_candidates=config.num_candidates,
    )

    stats = compute_data_statistics(config)
    write_json(config.data_statistics_path, stats)
    log_message(config.run_log_path, f"Wrote data statistics to {config.data_statistics_path}")


def _iter_raw_train(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            return

        if _looks_like_train_header(first):
            header = [_clean_header_name(name) for name in first]
            context_idx = _find_column(header, ["context"])
            response_idx = _find_column(header, ["utterance", "response"])
            label_idx = _find_column(header, ["label"])
        else:
            context_idx, response_idx, label_idx = 0, 1, 2
            record = _raw_train_row_to_record(first, context_idx, response_idx, label_idx)
            if record is not None:
                yield record

        for row in reader:
            record = _raw_train_row_to_record(row, context_idx, response_idx, label_idx)
            if record is not None:
                yield record


def _raw_train_row_to_record(
    row: Sequence[str],
    context_idx: int,
    response_idx: int,
    label_idx: int,
) -> Optional[Dict[str, object]]:
    if len(row) <= max(context_idx, response_idx, label_idx):
        return None
    try:
        label = int(float(row[label_idx]))
    except ValueError:
        return None
    if label not in (0, 1):
        return None
    return {
        "context": normalize_text(row[context_idx]),
        "response": normalize_text(row[response_idx]),
        "label": label,
    }


def _prepare_ranking_split(
    config: PartOneConfig,
    raw_path: Path,
    output_path: Path,
    dev_output_path: Path,
    main_size: int,
    dev_size: int,
    seed: int,
    split_name: str,
    num_candidates: int,
) -> None:
    records = list(_iter_raw_ranking(raw_path, num_candidates))
    if not records:
        raise ValueError(f"No ranking records were read from {raw_path}")

    rng = random.Random(seed)
    rng.shuffle(records)

    dev_records = records[: min(dev_size, len(records))]
    main_records = records[dev_size : dev_size + min(main_size, max(0, len(records) - dev_size))]

    _write_ranking_records(
        dev_output_path,
        _shuffle_candidates(dev_records, seed + 10, split_name, num_candidates),
        num_candidates,
    )
    _write_ranking_records(
        output_path,
        _shuffle_candidates(main_records, seed + 20, split_name, num_candidates),
        num_candidates,
    )
    log_message(
        config.run_log_path,
        f"Wrote {len(main_records)} {split_name} ranking rows to {output_path}",
    )
    log_message(
        config.run_log_path,
        f"Wrote {len(dev_records)} {split_name} dev ranking rows to {dev_output_path}",
    )


def _iter_raw_ranking(path: Path, num_candidates: int) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            return

        if _looks_like_ranking_header(first):
            header = [_clean_header_name(name) for name in first]
            context_idx = _find_column(header, ["context"])
            positive_idx = _find_column(header, ["ground_truth", "utterance", "response"])
            distractor_indices = [
                idx for idx in range(len(header)) if idx not in (context_idx, positive_idx)
            ]
        else:
            context_idx, positive_idx = 0, 1
            distractor_indices = list(range(2, len(first)))
            record = _raw_ranking_row_to_record(
                first, context_idx, positive_idx, distractor_indices, num_candidates
            )
            if record is not None:
                yield record

        for row in reader:
            record = _raw_ranking_row_to_record(
                row, context_idx, positive_idx, distractor_indices, num_candidates
            )
            if record is not None:
                yield record


def _raw_ranking_row_to_record(
    row: Sequence[str],
    context_idx: int,
    positive_idx: int,
    distractor_indices: Sequence[int],
    num_candidates: int,
) -> Optional[Dict[str, object]]:
    if len(row) <= max([context_idx, positive_idx] + list(distractor_indices)):
        return None

    positive = normalize_text(row[positive_idx])
    distractors = [normalize_text(row[idx]) for idx in distractor_indices if idx < len(row)]
    candidates = [positive] + distractors
    candidates = [candidate for candidate in candidates if candidate]
    if len(candidates) < num_candidates:
        return None
    return {
        "context": normalize_text(row[context_idx]),
        "candidates": candidates[:num_candidates],
    }


def _shuffle_candidates(
    records: Sequence[Dict[str, object]],
    seed: int,
    split_name: str,
    num_candidates: int,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    output_rows: List[Dict[str, object]] = []
    for idx, record in enumerate(records):
        candidates = list(record["candidates"])
        indexed = list(enumerate(candidates))
        rng.shuffle(indexed)

        label = next(new_idx for new_idx, (old_idx, _) in enumerate(indexed) if old_idx == 0)
        row: Dict[str, object] = {
            "sample_id": f"{split_name}_{idx:06d}",
            "context": record["context"],
            "label": label,
        }
        for candidate_idx in range(num_candidates):
            row[f"candidate_{candidate_idx}"] = indexed[candidate_idx][1]
        output_rows.append(row)
    return output_rows


def _reservoir_sample(
    records: Iterable[Dict[str, object]],
    sample_size: int,
    seed: int,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    reservoir: List[Dict[str, object]] = []
    for seen, record in enumerate(records):
        if len(reservoir) < sample_size:
            reservoir.append(record)
            continue
        replace_idx = rng.randint(0, seen)
        if replace_idx < sample_size:
            reservoir[replace_idx] = record
    rng.shuffle(reservoir)
    return reservoir


def _write_train_records(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    write_dicts_csv(path, rows, ["context", "response", "label"])


def _write_ranking_records(
    path: Path,
    rows: Sequence[Dict[str, object]],
    num_candidates: int,
) -> None:
    fieldnames = ["sample_id", "context"] + [
        f"candidate_{idx}" for idx in range(num_candidates)
    ] + ["label"]
    ordered_rows = []
    for row in rows:
        ordered = {field: row[field] for field in fieldnames}
        ordered_rows.append(ordered)
    write_dicts_csv(path, ordered_rows, fieldnames)


def compute_data_statistics(config: PartOneConfig) -> Dict[str, object]:
    return {
        "train_100k": _train_statistics(config.train_path),
        "valid_5k": _ranking_statistics(config.valid_path, config.num_candidates),
        "test_5k": _ranking_statistics(config.test_path, config.num_candidates),
    }


def _train_statistics(path: Path) -> Dict[str, object]:
    rows = _read_dict_rows(path)
    context_lengths = []
    response_lengths = []
    vocab = set()
    vocab_no_stop = set()
    labels = {0: 0, 1: 0}

    for row in rows:
        context_tokens = word_tokens(row["context"])
        response_tokens = word_tokens(row["response"])
        context_lengths.append(len(context_tokens))
        response_lengths.append(len(response_tokens))
        vocab.update(context_tokens)
        vocab.update(response_tokens)
        vocab_no_stop.update(remove_stopwords(context_tokens))
        vocab_no_stop.update(remove_stopwords(response_tokens))
        labels[int(row["label"])] += 1

    return {
        "num_rows": len(rows),
        "positive_rows": labels[1],
        "negative_rows": labels[0],
        "avg_context_tokens": _mean(context_lengths),
        "avg_response_tokens": _mean(response_lengths),
        "vocab_size": len(vocab),
        "vocab_size_without_stopwords": len(vocab_no_stop),
    }


def _ranking_statistics(path: Path, num_candidates: int) -> Dict[str, object]:
    rows = _read_dict_rows(path)
    context_lengths = []
    candidate_lengths = []
    ranks_vocab = set()

    for row in rows:
        context_tokens = word_tokens(row["context"])
        context_lengths.append(len(context_tokens))
        ranks_vocab.update(context_tokens)
        for idx in range(num_candidates):
            tokens = word_tokens(row[f"candidate_{idx}"])
            candidate_lengths.append(len(tokens))
            ranks_vocab.update(tokens)

    return {
        "num_rows": len(rows),
        "candidates_per_context": num_candidates,
        "avg_context_tokens": _mean(context_lengths),
        "avg_candidate_tokens": _mean(candidate_lengths),
        "vocab_size": len(ranks_vocab),
    }


def _read_dict_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def _mean(values: Sequence[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _looks_like_train_header(row: Sequence[str]) -> bool:
    joined = ",".join(_clean_header_name(item) for item in row)
    return "context" in joined and "label" in joined


def _looks_like_ranking_header(row: Sequence[str]) -> bool:
    joined = ",".join(_clean_header_name(item) for item in row)
    return "context" in joined and ("ground_truth" in joined or "utterance" in joined)


def _clean_header_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _find_column(header: Sequence[str], candidates: Sequence[str]) -> int:
    for candidate in candidates:
        for idx, name in enumerate(header):
            if candidate in name:
                return idx
    raise ValueError(f"Could not find any of {candidates} in header: {header}")
