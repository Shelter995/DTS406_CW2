from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PartTwoConfig:
    """Fixed configuration for the Part Two summarization and RAG experiment."""

    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])

    seed: int = 42
    initial_sample_size: int = 1_500
    sample_increment: int = 500
    final_sample_size: int = 1_000

    max_input_length: int = 512
    max_new_tokens: int = 120
    min_new_tokens: int = 30
    num_beams: int = 4
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    do_sample: bool = False

    chunk_target_words: int = 180
    chunk_overlap_words: int = 40
    retriever_top_k: int = 2
    qualitative_example_count: int = 5

    prefer_bf16: bool = True
    allow_fp16_fallback: bool = True

    data_dir: Path = field(init=False)
    raw_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    log_dir: Path = field(init=False)

    generator_model_dir: Path = field(init=False)
    retriever_model_dir: Path = field(init=False)

    cnn_archive_path: Path = field(init=False)
    dailymail_archive_path: Path = field(init=False)

    sampled_path: Path = field(init=False)
    final_dataset_path: Path = field(init=False)
    rag_chunks_path: Path = field(init=False)
    rag_retrievals_path: Path = field(init=False)
    filtered_out_path: Path = field(init=False)

    generation_results_path: Path = field(init=False)
    rouge_results_path: Path = field(init=False)
    final_metrics_path: Path = field(init=False)
    qualitative_examples_path: Path = field(init=False)
    data_statistics_path: Path = field(init=False)
    run_log_path: Path = field(init=False)
    figures_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir = self.root_dir / "data" / "part_two"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        self.output_dir = self.root_dir / "outputs" / "part_two"
        self.results_dir = self.output_dir / "results"
        self.log_dir = self.output_dir / "logs"
        self.figures_dir = self.results_dir / "figures"

        self.generator_model_dir = self.root_dir / "models" / "flan-t5-base"
        self.retriever_model_dir = self.root_dir / "models" / "all-MiniLM-L6-v2"

        self.cnn_archive_path = self.raw_dir / "cnn_stories.tgz"
        self.dailymail_archive_path = self.raw_dir / "dailymail_stories.tgz"

        self.sampled_path = self.processed_dir / "sampled_1500.csv"
        self.final_dataset_path = self.processed_dir / "final_1000.csv"
        self.rag_chunks_path = self.processed_dir / "rag_chunks.csv"
        self.rag_retrievals_path = self.processed_dir / "rag_retrievals.csv"
        self.filtered_out_path = self.processed_dir / "filtered_out.csv"

        self.generation_results_path = self.results_dir / "generation_results.csv"
        self.rouge_results_path = self.results_dir / "rouge_results.csv"
        self.final_metrics_path = self.results_dir / "final_metrics.json"
        self.qualitative_examples_path = self.results_dir / "qualitative_examples.csv"
        self.data_statistics_path = self.results_dir / "data_statistics.json"
        self.run_log_path = self.log_dir / "run.log"


def get_config() -> PartTwoConfig:
    return PartTwoConfig()
