from datetime import datetime
from pathlib import Path

from part_one.utils.io_utils import ensure_parent


def log_message(log_path: Path, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    ensure_parent(log_path)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
