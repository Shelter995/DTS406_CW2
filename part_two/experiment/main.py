from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from part_two.experiment.config import get_config
from part_two.experiment.evaluate import evaluate_summaries
from part_two.experiment.generate import generate_summaries
from part_two.utils.data_prepare import prepare_part_two_data
from part_two.utils.log_utils import log_message
from part_two.utils.seed_utils import set_global_seed


def main() -> None:
    config = get_config()
    set_global_seed(config.seed)

    log_message(config.run_log_path, "Starting Part Two summarization pipeline.")
    prepare_part_two_data(config)
    generate_summaries(config)
    evaluate_summaries(config)
    log_message(config.run_log_path, "Part Two pipeline finished.")


if __name__ == "__main__":
    main()
