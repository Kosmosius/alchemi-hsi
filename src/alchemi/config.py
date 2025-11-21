"""Shared configuration helpers for runtime settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(slots=True)
class RuntimeConfig:
    """Minimal runtime configuration shared across training entrypoints."""

    seed: int = 42
    device: str = "auto"
    dtype: str = "float32"
    amp_dtype: str | None = None
    deterministic: bool = False

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | None, fallback: Mapping[str, Any] | None = None
    ) -> "RuntimeConfig":
        raw: dict[str, Any] = {}
        if fallback:
            raw.update(fallback)
        if data:
            raw.update(data)
        return cls(
            seed=int(raw.get("seed", cls.seed)),
            device=str(raw.get("device", cls.device)),
            dtype=str(raw.get("dtype", cls.dtype)),
            amp_dtype=str(raw["amp_dtype"]) if raw.get("amp_dtype") is not None else None,
            deterministic=bool(raw.get("deterministic", cls.deterministic)),
        )

    def with_seed(self, seed: int | None) -> "RuntimeConfig":
        if seed is None:
            return self
        return RuntimeConfig(
            seed=seed,
            device=self.device,
            dtype=self.dtype,
            amp_dtype=self.amp_dtype,
            deterministic=self.deterministic,
        )

    @property
    def torch_device(self) -> torch.device:
        return select_device(self.device)

    @property
    def torch_dtype(self) -> torch.dtype:
        return resolve_dtype(self.dtype)

    @property
    def torch_amp_dtype(self) -> torch.dtype | None:
        if self.amp_dtype is None:
            return None
        return resolve_amp_dtype(self.amp_dtype)


def select_device(preference: str) -> torch.device:
    """Resolve a device preference string to a ``torch.device`` instance."""

    normalized = preference.lower()
    if normalized in {"auto", "cuda_if_available"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def resolve_dtype(name: str) -> torch.dtype:
    """Map a string representation to a torch dtype."""

    normalized = name.lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "double": torch.float64,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[normalized]


def resolve_amp_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported amp dtype: {name}")
    return mapping[normalized]

