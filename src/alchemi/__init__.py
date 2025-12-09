# TODO: migrated from legacy structure - reconcile with new design.
"""ALCHEMI: physics-aware any-sensor hyperspectral foundation model.

This package bundles the reusable library code for ingestion, physics, modeling,
and evaluation; the detailed system specification lives in ``docs/design/alchemi_design.tex``.
Refer to that design document for the authoritative description of contracts,
architecture, and evaluation scope.
"""

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
from .version import __version__

__all__ = [
    "__version__",
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
        "__version__",
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
