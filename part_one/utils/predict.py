from pathlib import Path
from typing import Dict, Iterable, List

from part_one.utils.io_utils import write_dicts_csv


def save_predictions(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        write_dicts_csv(path, [], [])
        return
    write_dicts_csv(path, rows, list(rows[0].keys()))


def save_error_examples(
    path: Path,
    prediction_rows: Iterable[Dict[str, object]],
    limit: int = 10,
) -> None:
    errors: List[Dict[str, object]] = []
    for row in prediction_rows:
        if int(row["correct_rank"]) == 1:
            continue

        label = int(row["label"])
        predicted_index = int(row["predicted_index"])
        errors.append(
            {
                "sample_id": row["sample_id"],
                "context": row["context"],
                "correct_rank": row["correct_rank"],
                "correct_candidate": row[f"candidate_{label}"],
                "predicted_candidate": row[f"candidate_{predicted_index}"],
                "correct_score": row[f"score_{label}"],
                "predicted_score": row[f"score_{predicted_index}"],
            }
        )

    errors.sort(key=lambda item: int(item["correct_rank"]), reverse=True)
    fieldnames = [
        "sample_id",
        "context",
        "correct_rank",
        "correct_candidate",
        "predicted_candidate",
        "correct_score",
        "predicted_score",
    ]
    write_dicts_csv(path, errors[:limit], fieldnames)

