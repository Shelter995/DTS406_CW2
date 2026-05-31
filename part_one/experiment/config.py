from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PartOneConfig:
    """Fixed configuration for the Part One response selection experiment."""

    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])

    seed: int = 42
    num_candidates: int = 10
    train_sample_size: int = 100_000
    train_dev_size: int = 1_000
    valid_sample_size: int = 5_000
    valid_dev_size: int = 500
    test_sample_size: int = 5_000
    test_dev_size: int = 500

    max_length: int = 256
    train_batch_size: int = 32
    eval_batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1

    prefer_bf16: bool = True
    allow_fp16_fallback: bool = True
    metric_for_best_model: str = "mrr"

    error_example_limit: int = 10

    data_dir: Path = field(init=False)
    raw_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    model_dir: Path = field(init=False)

    raw_train_path: Path = field(init=False)
    raw_valid_path: Path = field(init=False)
    raw_test_path: Path = field(init=False)

    train_path: Path = field(init=False)
    train_dev_path: Path = field(init=False)
    valid_path: Path = field(init=False)
    valid_dev_path: Path = field(init=False)
    test_path: Path = field(init=False)
    test_dev_path: Path = field(init=False)

    validation_metrics_path: Path = field(init=False)
    test_metrics_path: Path = field(init=False)
    test_predictions_path: Path = field(init=False)
    error_examples_path: Path = field(init=False)
    data_statistics_path: Path = field(init=False)
    run_log_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir = self.root_dir / "data" / "part_one"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.results_dir = self.data_dir / "results"

        self.output_dir = self.root_dir / "outputs" / "part_one"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"

        self.model_dir = self.root_dir / "models" / "MiniLM-L12-H384-uncased"

        self.raw_train_path = self.raw_dir / "train.csv"
        self.raw_valid_path = self.raw_dir / "valid.csv"
        self.raw_test_path = self.raw_dir / "test.csv"

        self.train_path = self.processed_dir / "train_100k.csv"
        self.train_dev_path = self.processed_dir / "train_dev_1k.csv"
        self.valid_path = self.processed_dir / "valid_5k.csv"
        self.valid_dev_path = self.processed_dir / "valid_dev_500.csv"
        self.test_path = self.processed_dir / "test_5k.csv"
        self.test_dev_path = self.processed_dir / "test_dev_500.csv"

        self.validation_metrics_path = self.results_dir / "validation_metrics.csv"
        self.test_metrics_path = self.results_dir / "test_metrics.json"
        self.test_predictions_path = self.results_dir / "test_predictions.csv"
        self.error_examples_path = self.results_dir / "error_examples.csv"
        self.data_statistics_path = self.results_dir / "data_statistics.json"
        self.run_log_path = self.log_dir / "run.log"


def get_config() -> PartOneConfig:
    return PartOneConfig()
