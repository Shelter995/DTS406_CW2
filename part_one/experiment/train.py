from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from part_one.experiment.config import PartOneConfig
from part_one.experiment.dataset import PairClassificationDataset
from part_one.experiment.evaluate import evaluate_model
from part_one.utils.io_utils import ensure_dir, write_dicts_csv
from part_one.utils.log_utils import log_message


def train_model(config: PartOneConfig) -> Path:
    ensure_dir(config.checkpoint_dir)
    ensure_dir(config.log_dir)
    log_message(config.run_log_path, f"Loading tokenizer and model from {config.model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(config.model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(config.model_dir),
        num_labels=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = PairClassificationDataset(
        config.train_path,
        tokenizer,
        config.max_length,
    )
    log_message(
        config.run_log_path,
        f"Training rows: {len(train_dataset)} | train batch size: {config.train_batch_size}",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_update_steps = (
        len(train_loader) * config.epochs
    ) // config.gradient_accumulation_steps
    warmup_steps = int(total_update_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    best_metric = -1.0
    best_checkpoint = config.checkpoint_dir / "best"
    validation_rows: List[Dict[str, object]] = []
    precision_mode = _precision_mode(config, device)
    scaler = _make_grad_scaler(precision_mode)
    log_message(
        config.run_log_path,
        f"Device: {device} | precision mode: {precision_mode} | epochs: {config.epochs}",
    )

    for epoch in range(1, config.epochs + 1):
        log_message(config.run_log_path, f"Starting epoch {epoch}/{config.epochs}")
        train_loss = _train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            scaler=scaler,
        )
        log_message(config.run_log_path, f"Finished epoch {epoch}; train loss={train_loss:.6f}")

        metrics, _ = evaluate_model(model, tokenizer, config, config.valid_path, device)
        row: Dict[str, object] = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "mrr": f"{metrics['mrr']:.6f}",
            "recall_at_1": f"{metrics['recall_at_1']:.6f}",
            "recall_at_2": f"{metrics['recall_at_2']:.6f}",
            "recall_at_5": f"{metrics['recall_at_5']:.6f}",
            "num_examples": int(metrics["num_examples"]),
        }
        validation_rows.append(row)
        write_dicts_csv(config.validation_metrics_path, validation_rows, list(row.keys()))

        current_metric = float(metrics[config.metric_for_best_model])
        if current_metric > best_metric:
            best_metric = current_metric
            model.save_pretrained(str(best_checkpoint))
            tokenizer.save_pretrained(str(best_checkpoint))
            log_message(
                config.run_log_path,
                f"Saved new best checkpoint to {best_checkpoint} with "
                f"{config.metric_for_best_model}={best_metric:.6f}",
            )

    return best_checkpoint


def _train_one_epoch(
    model,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    config: PartOneConfig,
    device: torch.device,
    scaler,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    autocast_context = _autocast_context(config, device)

    total_loss = 0.0
    update_count = 0

    for step, batch in enumerate(train_loader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
        with autocast_context:
            loss = model(**batch).loss
            loss = loss / config.gradient_accumulation_steps

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % config.gradient_accumulation_steps == 0:
            _optimizer_step(model, optimizer, scheduler, scaler)
            update_count += 1

        total_loss += loss.detach().float().item() * config.gradient_accumulation_steps
        if step == 1 or step % 100 == 0 or step == len(train_loader):
            print(
                f"  step {step}/{len(train_loader)} "
                f"loss={loss.detach().float().item() * config.gradient_accumulation_steps:.6f}",
                flush=True,
            )

    if len(train_loader) % config.gradient_accumulation_steps != 0:
        _optimizer_step(model, optimizer, scheduler, scaler)
        update_count += 1

    return total_loss / max(1, len(train_loader))


def _optimizer_step(model, optimizer, scheduler, scaler) -> None:
    if scaler.is_enabled():
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def _make_grad_scaler(precision_mode: str):
    enabled = precision_mode == "fp16"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _precision_mode(config: PartOneConfig, device: torch.device) -> str:
    if device.type != "cuda":
        return "none"
    if config.prefer_bf16 and torch.cuda.is_bf16_supported():
        return "bf16"
    if config.allow_fp16_fallback:
        return "fp16"
    return "none"


def _autocast_context(config: PartOneConfig, device: torch.device):
    precision_mode = _precision_mode(config, device)
    if precision_mode == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision_mode == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()
