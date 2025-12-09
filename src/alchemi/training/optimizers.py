"""Optimizer factory functions used by the trainer."""

from __future__ import annotations

from typing import Iterable

import torch

from alchemi.config.core import OptimizerConfig


def build_optimizer(
    params: Iterable[torch.nn.Parameter],
    cfg: OptimizerConfig,
    *,
    lr_override: float | None = None,
) -> torch.optim.Optimizer:
    lr = lr_override if lr_override is not None else cfg.lr
    if cfg.type == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
    if cfg.type == "sgd":
        momentum = cfg.momentum or 0.9
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unsupported optimizer type: {cfg.type}")


__all__ = ["build_optimizer"]
