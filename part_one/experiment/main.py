from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from part_one.experiment.config import get_config
from part_one.experiment.evaluate import evaluate_checkpoint
from part_one.experiment.train import train_model
from part_one.utils.data_prepare import prepare_part_one_data
from part_one.utils.log_utils import log_message
from part_one.utils.seed_utils import set_global_seed


def main() -> None:
    config = get_config()
    set_global_seed(config.seed)

    log_message(config.run_log_path, "Starting Part One response selection pipeline.")
    prepare_part_one_data(config)
    best_checkpoint = train_model(config)
    evaluate_checkpoint(
        config=config,
        checkpoint_dir=best_checkpoint,
        data_path=config.test_path,
        save_outputs=True,
    )
    log_message(config.run_log_path, "Part One pipeline finished.")


if __name__ == "__main__":
    main()
