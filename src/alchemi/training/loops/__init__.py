"""Lightweight training/evaluation loops shared across stages."""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable

import torch

from alchemi.training.logging import MetricLogger


def _default_logger(logger: MetricLogger | None) -> MetricLogger:
    return logger or MetricLogger(enable_wandb=False)


def train_epoch(
    dataloader: Iterable,
    step_fn: Callable[[dict], tuple[torch.Tensor, dict[str, float]]],
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    accumulation_steps: int = 1,
    logger: MetricLogger | None = None,
) -> dict[str, float]:
    """Iterate over a dataloader with gradient steps and metric logging."""

    logger = _default_logger(logger)
    optimizer.zero_grad(set_to_none=True)
    stats: dict[str, list[float]] = defaultdict(list)

    for step, batch in enumerate(dataloader, start=1):
        loss, metrics = step_fn(batch)
        (loss / accumulation_steps).backward()
        if step % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        metrics = {**metrics, "loss": float(loss.detach().cpu())}
        logger.log_train(metrics)
        for key, value in metrics.items():
            stats[key].append(value)

    return {k: float(sum(v) / max(len(v), 1)) for k, v in stats.items()}


def evaluate(
    dataloader: Iterable,
    step_fn: Callable[[dict], dict[str, float]],
    *,
    logger: MetricLogger | None = None,
) -> dict[str, float]:
    """Evaluation loop without gradient tracking."""

    logger = _default_logger(logger)
    stats: dict[str, list[float]] = defaultdict(list)

    with torch.no_grad():
        for batch in dataloader:
            metrics = step_fn(batch)
            logger.log_eval(metrics)
            for key, value in metrics.items():
                stats[key].append(float(value))

    return {k: float(sum(v) / max(len(v), 1)) for k, v in stats.items()}


__all__ = ["train_epoch", "evaluate"]
