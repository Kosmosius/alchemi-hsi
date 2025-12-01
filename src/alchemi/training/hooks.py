"""Hook utilities for training stages."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


@dataclass
class HookContext:
    epoch: int
    metrics: dict[str, float]


class CheckpointHook:
    def __init__(self, path: str | Path = "checkpoints") -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def __call__(self, model: torch.nn.Module, ctx: HookContext) -> None:
        filename = self.path / f"epoch{ctx.epoch}.pt"
        torch.save({"state_dict": model.state_dict(), "metrics": ctx.metrics}, filename)


class EarlyStoppingHook:
    def __init__(self, patience: int = 5, metric: str = "loss") -> None:
        self.patience = patience
        self.metric = metric
        self.best: float | None = None
        self.bad_epochs = 0

    def __call__(self, ctx: HookContext) -> bool:
        current = ctx.metrics.get(self.metric, float("inf"))
        if self.best is None or current < self.best:
            self.best = current
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class EvaluationHook:
    def __init__(self, evaluator: Callable[[], dict[str, float]]) -> None:
        self.evaluator = evaluator

    def __call__(self) -> dict[str, float]:
        return self.evaluator()


__all__ = ["HookContext", "CheckpointHook", "EarlyStoppingHook", "EvaluationHook"]
