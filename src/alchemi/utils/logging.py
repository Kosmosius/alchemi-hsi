from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter

import torch
from rich.logging import RichHandler


def get_logger(name: str = "alchemi") -> logging.Logger:
    """Return a Rich-configured logger for the project."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger(name)


@dataclass(slots=True)
class ThroughputStats:
    """Summary of per-step or aggregated throughput metrics."""

    tokens_per_s: float
    gb_per_s: float | None
    peak_mem_gb: float | None
    step_time_s: float
    tokens: int
    num_bytes: int | None

    def as_dict(self) -> dict[str, float]:
        """Return serialisable metrics with missing values replaced by zeros."""
        return {
            "tokens_per_s": float(self.tokens_per_s),
            "gb_per_s": float(self.gb_per_s) if self.gb_per_s is not None else 0.0,
            "peak_mem_gb": float(self.peak_mem_gb) if self.peak_mem_gb is not None else 0.0,
            "step_time_s": float(self.step_time_s),
            "tokens": float(self.tokens),
            "num_bytes": float(self.num_bytes) if self.num_bytes is not None else 0.0,
        }


class ThroughputMeter:
    """Lightweight helper to compute throughput and memory statistics."""

    def __init__(self, device: torch.device | str | None = None) -> None:
        self.device = torch.device(device) if device is not None else None
        self._start_time: float | None = None

    def _sync(self) -> None:
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self) -> None:
        """Mark the beginning of a measured section."""
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._sync()
        self._start_time = perf_counter()

    def stop(self, tokens: int, num_bytes: int | None = None) -> ThroughputStats:
        """Mark the end of a measured section and return throughput stats."""
        if self._start_time is None:
            raise RuntimeError("ThroughputMeter.stop() called before start().")
        self._sync()
        elapsed = perf_counter() - self._start_time
        self._start_time = None
        elapsed = max(elapsed, 1e-12)

        tokens_per_s = tokens / elapsed
        gb_per_s = num_bytes / (elapsed * 1e9) if num_bytes is not None else None
        peak_mem = None
        if self.device is not None and self.device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated(self.device) / 1e9

        return ThroughputStats(
            tokens_per_s=tokens_per_s,
            gb_per_s=gb_per_s,
            peak_mem_gb=peak_mem,
            step_time_s=elapsed,
            tokens=tokens,
            num_bytes=num_bytes,
        )

    def log(  # pragma: no cover - thin wrapper around logger
        self,
        logger: logging.Logger,
        step: int,
        stats: ThroughputStats,
        *,
        prefix: str = "train",
    ) -> None:
        """Log a standardised throughput line."""
        gb_per_s = f"{stats.gb_per_s:.2f}" if stats.gb_per_s is not None else "n/a"
        peak_mem = f"{stats.peak_mem_gb:.2f}" if stats.peak_mem_gb is not None else "n/a"
        logger.info(
            "%s step=%d tokens/s=%.1f gb/s=%s peak_mem=%sGB",
            prefix,
            step,
            stats.tokens_per_s,
            gb_per_s,
            peak_mem,
        )
