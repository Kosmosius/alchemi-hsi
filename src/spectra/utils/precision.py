from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import torch

PrecisionType = Literal["fp8", "bf16", "fp16", "fp32"]

_PRECISION_ALIASES: dict[str, PrecisionType] = {
    "fp8": "fp8",
    "bf16": "bf16",
    "fp16": "fp16",
    "fp32": "fp32",
}


def _has_transformer_engine() -> bool:
    try:

        return True
    except Exception:  # pragma: no cover - import guards
        return False


@contextmanager
def fp8_autocast() -> Iterator[None]:
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
        if self.target == "fp8" and self.enable_fp8 and _has_transformer_engine():
            return "fp8"
        if self.target == "fp8":
            return "bf16"
        return self.target

    def autocast_enabled(self) -> bool:
        return self.resolved_precision() in {"fp8", "bf16", "fp16"}

    def autocast_dtype(self) -> torch.dtype | None:
        match self.resolved_precision():
            case "bf16":
                return torch.bfloat16
            case "fp16":
                return torch.float16
            case _:
                return None


@contextmanager
def autocast(precision: PrecisionConfig) -> Iterator[None]:
    resolved = precision.resolved_precision()
    if resolved == "fp8":
        with fp8_autocast():
            yield
        return

    dtype = precision.autocast_dtype()
    enabled = precision.autocast_enabled() and dtype is not None and precision.device_type == "cuda"
    with torch.autocast(device_type=precision.device_type, dtype=dtype, enabled=enabled):
        yield


@contextmanager
def autocast_from_config(
    config: Mapping[str, object] | PrecisionConfig | None,
) -> Iterator[None]:
    if config is None:
        cfg = PrecisionConfig()
    elif isinstance(config, PrecisionConfig):
        cfg = config
    else:
        raw = str(config.get("precision", "bf16"))
        precision = _PRECISION_ALIASES.get(raw, "bf16")
        cfg = PrecisionConfig(target=precision)
    with autocast(cfg):
        yield


__all__ = [
    "PrecisionConfig",
    "PrecisionType",
    "autocast",
    "autocast_from_config",
    "fp8_autocast",
]
