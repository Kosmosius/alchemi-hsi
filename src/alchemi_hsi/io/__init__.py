"""IO utilities for the :mod:`alchemi_hsi` package."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

from .enmap import enmap_pixel, load_enmap_l1b

__all__ = ["enmap_pixel", "load_enmap_l1b"]

if find_spec("alchemi_hsi.io.emit") is not None:
    emit = import_module("alchemi_hsi.io.emit")
    __all__.extend(["emit_pixel", "load_emit_l1b"])
    emit_pixel = getattr(emit, "emit_pixel")
    load_emit_l1b = getattr(emit, "load_emit_l1b")
