"""Minimal metric logging helpers."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class MetricLogger:
    enable_wandb: bool = False
    history: Dict[str, list[float]] = field(default_factory=dict)

    def log_train(self, metrics: dict[str, float]) -> None:
        self._record(metrics)
        logging.info("train %s", metrics)
        self._log_wandb(metrics, prefix="train")

    def log_eval(self, metrics: dict[str, float]) -> None:
        self._record(metrics)
        logging.info("eval %s", metrics)
        self._log_wandb(metrics, prefix="eval")

    def _record(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.history.setdefault(key, []).append(value)

    def _log_wandb(self, metrics: dict[str, float], *, prefix: str) -> None:
        if not self.enable_wandb:
            return
        try:
            import wandb

            wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()})
        except Exception:
            # Optional dependency; ignore failures
            pass


__all__ = ["MetricLogger"]
