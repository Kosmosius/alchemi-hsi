"""Precision helpers with fp8/bf16/fp16 autocast toggles."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal, Optional

import torch

PrecisionType = Literal["fp8", "bf16", "fp16", "fp32"]


def _has_transformer_engine() -> bool:
    try:
        import transformer_engine.pytorch  # type: ignore

        return True
    except Exception:  # pragma: no cover - import guards
        return False


@contextmanager
def fp8_autocast() -> Iterator[None]:
    """Activate TransformerEngine fp8 autocast if available; otherwise no-op."""
    if not _has_transformer_engine():
        yield
        return

    from transformer_engine.pytorch import fp8_autocast as te_autocast  # type: ignore

    with te_autocast():
        yield


@dataclass
class PrecisionConfig:
    target: PrecisionType = "bf16"
    device_type: str = "cuda" if torch.cuda.is_available() else "cpu"
    enable_fp8: bool = False

    def resolved_precision(self) -> PrecisionType:
        """Return the actual precision that will be used."""
        if self.target == "fp8" and self.enable_fp8 and _has_transformer_engine():
            return "fp8"
        if self.target == "fp8":
            # Fall back when fp8 is requested but unavailable.
            return "bf16"
        return self.target

    def autocast_enabled(self) -> bool:
        return self.resolved_precision() in {"fp8", "bf16", "fp16"}

    def autocast_dtype(self) -> Optional[torch.dtype]:
        match self.resolved_precision():
            case "bf16":
                return torch.bfloat16
            case "fp16":
                return torch.float16
            case _:
                return None


@contextmanager
def autocast(precision: PrecisionConfig) -> Iterator[None]:
    """Autocast context that routes through fp8 / bf16 / fp16 when available."""
    resolved = precision.resolved_precision()
    if resolved == "fp8":
        with fp8_autocast():
            yield
        return

    dtype = precision.autocast_dtype()
    enabled = precision.autocast_enabled() and dtype is not None and precision.device_type == "cuda"
    with torch.autocast(device_type=precision.device_type, dtype=dtype, enabled=enabled):
        yield


__all__ = ["PrecisionConfig", "autocast", "fp8_autocast", "PrecisionType"]
