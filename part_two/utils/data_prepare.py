import random
import tarfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

from transformers import AutoTokenizer

from part_two.experiment.config import PartTwoConfig
from part_two.utils.io_utils import ensure_dir, write_dicts_csv, write_json
from part_two.utils.log_utils import log_message
from part_two.utils.rag_utils import (
    EmbeddingRetriever,
    build_summary_prompt,
    make_query,
    retrieve_top_chunks,
    sentence_aware_chunks,
)
from part_two.utils.text_utils import normalize_text, remove_stopwords, word_tokens


BASELINE_PROMPT = "Summarize the following news article in three concise sentences:"
RAG_PROMPT = "Summarize the following news passages in three concise sentences:"


def prepare_part_two_data(config: PartTwoConfig) -> None:
    """Prepare sampled CNN/DailyMail data and intra-document RAG retrieval files."""
    for path in [config.raw_dir, config.processed_dir, config.results_dir, config.log_dir]:
        ensure_dir(path)

    _check_raw_archives(config)
    tokenizer = AutoTokenizer.from_pretrained(str(config.generator_model_dir))
    retriever = EmbeddingRetriever(config.retriever_model_dir)

    sample_size = config.initial_sample_size
    while True:
        log_message(config.run_log_path, f"Sampling {sample_size} raw stories.")
        sampled = _reservoir_sample(_iter_story_examples(config), sample_size, config.seed)
        write_dicts_csv(
            config.sampled_path,
            sampled,
            ["id", "source", "article", "reference"],
        )

        prepared = _build_rag_files(config, sampled, tokenizer, retriever)
        valid_count = len(prepared["final_rows"])
        log_message(
            config.run_log_path,
            f"Valid examples after RAG length filtering: {valid_count}",
        )
        if valid_count >= config.final_sample_size:
            break
        sample_size += config.sample_increment
        log_message(
            config.run_log_path,
            f"Not enough valid examples; increasing sample size to {sample_size}.",
        )

    stats = compute_data_statistics(config.final_dataset_path)
    write_json(config.data_statistics_path, stats)
    log_message(config.run_log_path, f"Wrote Part Two data statistics to {config.data_statistics_path}")


def _build_rag_files(
    config: PartTwoConfig,
    sampled: Sequence[Dict[str, str]],
    tokenizer,
    retriever: EmbeddingRetriever,
) -> Dict[str, List[Dict[str, object]]]:
    final_rows: List[Dict[str, object]] = []
    chunk_rows: List[Dict[str, object]] = []
    retrieval_rows: List[Dict[str, object]] = []
    filtered_rows: List[Dict[str, object]] = []

    for index, example in enumerate(sampled):
        if index == 0 or (index + 1) % 100 == 0:
            log_message(
                config.run_log_path,
                f"Preparing RAG retrievals for example {index + 1}/{len(sampled)}",
            )

        article = normalize_text(example["article"])
        reference = normalize_text(example["reference"])
        query = make_query(article)
        chunks = sentence_aware_chunks(
            article,
            target_words=config.chunk_target_words,
            overlap_words=config.chunk_overlap_words,
        )
        selected = retrieve_top_chunks(
            query=query,
            chunks=chunks,
            retriever=retriever,
            top_k=config.retriever_top_k,
        )
        retrieved_text = normalize_text(" ".join(item["chunk"].text for item in selected))
        rag_prompt = build_summary_prompt(RAG_PROMPT, retrieved_text)
        rag_prompt_tokens = len(tokenizer(rag_prompt, add_special_tokens=True)["input_ids"])

        for chunk in chunks:
            chunk_rows.append(
                {
                    "id": example["id"],
                    "chunk_id": chunk.chunk_id,
                    "start_sentence": chunk.start_sentence,
                    "end_sentence": chunk.end_sentence,
                    "word_count": chunk.word_count,
                    "chunk_text": chunk.text,
                }
            )

        for rank, item in enumerate(selected, start=1):
            chunk = item["chunk"]
            retrieval_rows.append(
                {
                    "id": example["id"],
                    "retrieval_rank": rank,
                    "chunk_id": chunk.chunk_id,
                    "score": f"{float(item['score']):.8f}",
                    "chunk_text": chunk.text,
                }
            )

        if rag_prompt_tokens > config.max_input_length:
            filtered_rows.append(
                {
                    "id": example["id"],
                    "source": example["source"],
                    "rag_prompt_token_length": rag_prompt_tokens,
                    "reason": "rag_prompt_too_long",
                }
            )
            continue

        if len(final_rows) < config.final_sample_size:
            final_rows.append(
                {
                    "id": example["id"],
                    "source": example["source"],
                    "article": article,
                    "reference": reference,
                    "query": query,
                    "retrieved_chunks": retrieved_text,
                    "rag_prompt_token_length": rag_prompt_tokens,
                }
            )

    write_dicts_csv(
        config.final_dataset_path,
        final_rows,
        [
            "id",
            "source",
            "article",
            "reference",
            "query",
            "retrieved_chunks",
            "rag_prompt_token_length",
        ],
    )
    write_dicts_csv(
        config.rag_chunks_path,
        chunk_rows,
        ["id", "chunk_id", "start_sentence", "end_sentence", "word_count", "chunk_text"],
    )
    write_dicts_csv(
        config.rag_retrievals_path,
        retrieval_rows,
        ["id", "retrieval_rank", "chunk_id", "score", "chunk_text"],
    )
    write_dicts_csv(
        config.filtered_out_path,
        filtered_rows,
        ["id", "source", "rag_prompt_token_length", "reason"],
    )

    return {
        "final_rows": final_rows,
        "chunk_rows": chunk_rows,
        "retrieval_rows": retrieval_rows,
        "filtered_rows": filtered_rows,
    }


def compute_data_statistics(final_dataset_path: Path) -> Dict[str, object]:
    import csv

    with final_dataset_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.DictReader(f))

    article_lengths: List[int] = []
    reference_lengths: List[int] = []
    compression_ratios: List[float] = []
    vocab = set()
    vocab_no_stop = set()

    for row in rows:
        article_tokens = word_tokens(row["article"])
        reference_tokens = word_tokens(row["reference"])
        article_lengths.append(len(article_tokens))
        reference_lengths.append(len(reference_tokens))
        if article_tokens:
            compression_ratios.append(len(reference_tokens) / len(article_tokens))
        vocab.update(article_tokens)
        vocab.update(reference_tokens)
        vocab_no_stop.update(remove_stopwords(article_tokens))
        vocab_no_stop.update(remove_stopwords(reference_tokens))

    return {
        "num_examples": len(rows),
        "avg_article_words": _mean(article_lengths),
        "median_article_words": _median(article_lengths),
        "avg_reference_words": _mean(reference_lengths),
        "median_reference_words": _median(reference_lengths),
        "avg_compression_ratio": _mean(compression_ratios),
        "median_compression_ratio": _median(compression_ratios),
        "vocab_size": len(vocab),
        "vocab_size_without_stopwords": len(vocab_no_stop),
    }


def _iter_story_examples(config: PartTwoConfig) -> Iterator[Dict[str, str]]:
    yield from _iter_archive(config.cnn_archive_path, "cnn")
    yield from _iter_archive(config.dailymail_archive_path, "dailymail")


def _iter_archive(archive_path: Path, source: str) -> Iterator[Dict[str, str]]:
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".story"):
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            text = extracted.read().decode("utf-8", errors="replace")
            article, highlights = _parse_story(text)
            if not article or not highlights:
                continue
            story_id = f"{source}_{Path(member.name).stem}"
            yield {
                "id": story_id,
                "source": source,
                "article": article,
                "reference": highlights,
            }


def _parse_story(text: str) -> Tuple[str, str]:
    article_lines: List[str] = []
    highlights: List[str] = []
    lines = [line.strip() for line in text.splitlines()]
    in_highlight = False

    for line in lines:
        if not line:
            continue
        if line.lower().startswith("@highlight"):
            in_highlight = True
            continue
        if in_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    article = normalize_text(" ".join(article_lines))
    reference = normalize_text(" ".join(highlights))
    return article, reference


def _reservoir_sample(
    records: Iterable[Dict[str, str]],
    sample_size: int,
    seed: int,
) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    reservoir: List[Dict[str, str]] = []
    for seen, record in enumerate(records):
        if len(reservoir) < sample_size:
            reservoir.append(record)
            continue
        replace_idx = rng.randint(0, seen)
        if replace_idx < sample_size:
            reservoir[replace_idx] = record
    rng.shuffle(reservoir)
    return reservoir


def _check_raw_archives(config: PartTwoConfig) -> None:
    missing = [
        path for path in [config.cnn_archive_path, config.dailymail_archive_path] if not path.exists()
    ]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing CNN/DailyMail raw archive(s): {missing_text}")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: Sequence[float]) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    midpoint = len(values) // 2
    if len(values) % 2:
        return float(values[midpoint])
    return (values[midpoint - 1] + values[midpoint]) / 2
