from pathlib import Path
from typing import Dict, List

from part_two.utils.io_utils import read_dicts_csv


class SummarizationDataset:
    def __init__(self, csv_path: Path) -> None:
        self.rows = read_dicts_csv(csv_path)
        if not self.rows:
            raise ValueError(f"No summarization rows found in {csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def records(self) -> List[Dict[str, str]]:
        return self.rows

