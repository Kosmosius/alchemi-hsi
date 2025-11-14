"""Deprecated compatibility shim for the former :mod:`alchemi_hsi` package."""

from __future__ import annotations

import sys
import warnings
from importlib import import_module
from types import ModuleType

__version__ = "0.2.0"

warnings.warn(
    "`alchemi_hsi` is deprecated; please import from `alchemi` instead.",
    DeprecationWarning,
    stacklevel=2,
)

_alchemi: ModuleType = import_module("alchemi")
__all__ = ["__version__"]

_exported = getattr(_alchemi, "__all__", None)
if _exported:
    __all__.extend(_exported)
    globals().update({name: getattr(_alchemi, name) for name in _exported})


def __getattr__(name: str):
    try:
        return getattr(_alchemi, name)
    except AttributeError as exc:  # pragma: no cover - mirrors stdlib behaviour
        raise AttributeError(f"module 'alchemi_hsi' has no attribute {name}") from exc


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(dir(_alchemi)))


def _alias_module(alias: str, target: str) -> ModuleType:
    module = import_module(target)
    sys.modules.setdefault(alias, module)
    return module


io = _alias_module("alchemi_hsi.io", "alchemi.data.io")
__all__.append("io")

srf = _alias_module("alchemi_hsi.srf", "alchemi.srf")
__all__.append("srf")


def _alias_submodules(pairs: dict[str, str]) -> None:
    for alias, target in pairs.items():
        sys.modules.setdefault(alias, import_module(target))


_alias_submodules(
    {
        "alchemi_hsi.io.avirisng": "alchemi.data.io.avirisng",
        "alchemi_hsi.io.emit": "alchemi.data.io.emit",
        "alchemi_hsi.io.enmap": "alchemi.data.io.enmap",
        "alchemi_hsi.io.hytes": "alchemi.data.io.hytes",
        "alchemi_hsi.io.splib": "alchemi.data.io.splib",
        "alchemi_hsi.srf.avirisng": "alchemi.srf.avirisng",
        "alchemi_hsi.srf.emit": "alchemi.srf.emit",
        "alchemi_hsi.srf.enmap": "alchemi.srf.enmap",
        "alchemi_hsi.srf.fallback": "alchemi.srf.fallback",
        "alchemi_hsi.srf.hytes": "alchemi.srf.hytes",
        "alchemi_hsi.srf.resample": "alchemi.srf.resample",
    }
)
