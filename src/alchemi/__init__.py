"""Top-level package exports with lazy submodule loading."""

from __future__ import annotations

import importlib
from typing import Any

from .types import (
    QuantityKind,
    Sample,
    SampleMeta,
    Spectrum,
    SpectrumKind,
    SRFMatrix,
    ValueUnits,
    WavelengthGrid,
)

__all__ = [
    "SRFMatrix",
    "Sample",
    "SampleMeta",
    "Spectrum",
    "QuantityKind",
    "ValueUnits",
    "SpectrumKind",
    "WavelengthGrid",
    "align",
    "data",
    "eval",
    "io",
    "models",
    "physics",
    "srf",
    "tokens",
    "training",
    "utils",
]

_SUBMODULES = {
    name
    for name in __all__
    if name
    not in {
        "SRFMatrix",
        "Sample",
        "SampleMeta",
        "Spectrum",
        "QuantityKind",
        "ValueUnits",
        "SpectrumKind",
        "WavelengthGrid",
    }
}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
