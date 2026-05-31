import csv
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class PairClassificationDataset(Dataset):
    """Training dataset for context-response binary classification."""

    def __init__(self, csv_path: Path, tokenizer, max_length: int) -> None:
        self.rows = _read_rows(csv_path)
        if not self.rows:
            raise ValueError(f"No training rows were found in {csv_path}")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.rows[index]
        encoded = self.tokenizer(
            row["context"],
            row["response"],
            max_length=self.max_length,
            truncation="longest_first",
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return item


class RankingDataset:
    """Evaluation dataset for 1-of-10 candidate ranking."""

    def __init__(self, csv_path: Path, num_candidates: int) -> None:
        self.rows = _read_rows(csv_path)
        if not self.rows:
            raise ValueError(f"No ranking rows were found in {csv_path}")
        self.num_candidates = num_candidates

    def __len__(self) -> int:
        return len(self.rows)

    def records(self) -> List[Dict[str, object]]:
        records = []
        for row in self.rows:
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "context": row["context"],
                    "candidates": [
                        row[f"candidate_{idx}"] for idx in range(self.num_candidates)
                    ],
                    "label": int(row["label"]),
                }
            )
        return records


def _read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))
