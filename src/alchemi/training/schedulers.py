"""Scheduler utilities."""
from __future__ import annotations

import math

import torch

from alchemi.config.core import SchedulerConfig


class WarmupCosine(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, max_steps: int, min_lr: float = 0.0):
        self.warmup_steps = max(1, warmup_steps)
        self.max_steps = max(self.warmup_steps + 1, max_steps)
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:  # pragma: no cover - math only
        step = max(1, self.last_epoch)
        warmup = min(1.0, step / float(self.warmup_steps))
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return [base_lr * warmup * cosine + self.min_lr for base_lr in self.base_lrs]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: SchedulerConfig,
    max_epochs: int | None = None,
) -> torch.optim.lr_scheduler._LRScheduler:
    if cfg.type == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    max_steps = max_epochs if max_epochs is not None else cfg.max_epochs or 1
    return WarmupCosine(
        optimizer,
        warmup_steps=cfg.warmup_steps,
        max_steps=max_steps,
        min_lr=cfg.min_lr,
    )


__all__ = ["build_scheduler", "WarmupCosine"]
