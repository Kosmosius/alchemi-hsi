"""IO utilities for the :mod:`alchemi_hsi` package."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

from .avirisng import avirisng_pixel, load_avirisng_l1b
from .enmap import enmap_pixel, load_enmap_l1b
from .splib import load_splib, SPLIBCatalog

__all__ = [
    "avirisng_pixel",
    "enmap_pixel",
    "load_avirisng_l1b",
    "load_enmap_l1b",
    "SPLIBCatalog",
]

if find_spec("alchemi_hsi.io.emit") is not None:
    emit = import_module("alchemi_hsi.io.emit")
    __all__.extend(["emit_pixel", "load_emit_l1b"])
    emit_pixel = getattr(emit, "emit_pixel")
    load_emit_l1b = getattr(emit, "load_emit_l1b")

if find_spec("alchemi_hsi.io.hytes") is not None:
    hytes = import_module("alchemi_hsi.io.hytes")
    __all__.extend(["hytes_pixel_bt", "load_hytes_l1b_bt"])
    hytes_pixel_bt = getattr(hytes, "hytes_pixel_bt")
    load_hytes_l1b_bt = getattr(hytes, "load_hytes_l1b_bt")
