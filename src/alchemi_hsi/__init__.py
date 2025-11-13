"""ALCHEMI hyperspectral imaging public interface."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from types import ModuleType

from . import io

__version__ = "0.2.0"

_alchemi: ModuleType = import_module("alchemi")
_exported: Iterable[str] = getattr(_alchemi, "__all__", ())

# Mirror the alchemi namespace for backwards compatibility while exposing the package version.
alchemi = _alchemi

for _name in _exported:
    globals()[_name] = getattr(_alchemi, _name)

__all__ = ["__version__", "alchemi", "io", *_exported]
