from contextlib import nullcontext
from typing import Dict, List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from part_two.experiment.config import PartTwoConfig
from part_two.experiment.dataset import SummarizationDataset
from part_two.utils.data_prepare import BASELINE_PROMPT, RAG_PROMPT
from part_two.utils.io_utils import write_dicts_csv
from part_two.utils.log_utils import log_message
from part_two.utils.rag_utils import build_summary_prompt
from part_two.utils.text_utils import normalize_text


def generate_summaries(config: PartTwoConfig) -> None:
    dataset = SummarizationDataset(config.final_dataset_path)
    records = dataset.records()

    tokenizer = AutoTokenizer.from_pretrained(str(config.generator_model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(config.generator_model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    precision_mode = _precision_mode(config, device)
    log_message(
        config.run_log_path,
        f"Generating summaries for {len(records)} examples on {device} with {precision_mode}.",
    )

    rows: List[Dict[str, object]] = []
    for index, record in enumerate(records):
        if index == 0 or (index + 1) % 50 == 0:
            log_message(config.run_log_path, f"Generating example {index + 1}/{len(records)}")

        baseline_prompt = build_summary_prompt(BASELINE_PROMPT, record["article"])
        rag_prompt = build_summary_prompt(RAG_PROMPT, record["retrieved_chunks"])

        baseline_summary = _generate_one(model, tokenizer, config, baseline_prompt, device)
        rag_summary = _generate_one(model, tokenizer, config, rag_prompt, device)

        rows.append(
            {
                "id": record["id"],
                "source": record["source"],
                "article": record["article"],
                "reference": record["reference"],
                "retrieved_chunks": record["retrieved_chunks"],
                "baseline_summary": baseline_summary,
                "rag_summary": rag_summary,
            }
        )

    write_dicts_csv(
        config.generation_results_path,
        rows,
        [
            "id",
            "source",
            "article",
            "reference",
            "retrieved_chunks",
            "baseline_summary",
            "rag_summary",
        ],
    )
    log_message(config.run_log_path, f"Wrote summaries to {config.generation_results_path}")


def _generate_one(model, tokenizer, config: PartTwoConfig, prompt: str, device: torch.device) -> str:
    encoded = tokenizer(
        prompt,
        max_length=config.max_input_length,
        truncation=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        with _autocast_context(config, device):
            output_ids = model.generate(
                **encoded,
                max_new_tokens=config.max_new_tokens,
                min_new_tokens=config.min_new_tokens,
                num_beams=config.num_beams,
                length_penalty=config.length_penalty,
                early_stopping=True,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                do_sample=config.do_sample,
            )
    return normalize_text(tokenizer.decode(output_ids[0], skip_special_tokens=True))


def _precision_mode(config: PartTwoConfig, device: torch.device) -> str:
    if device.type != "cuda":
        return "none"
    if config.prefer_bf16 and torch.cuda.is_bf16_supported():
        return "bf16"
    if config.allow_fp16_fallback:
        return "fp16"
    return "none"


def _autocast_context(config: PartTwoConfig, device: torch.device):
    mode = _precision_mode(config, device)
    if mode == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if mode == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()

