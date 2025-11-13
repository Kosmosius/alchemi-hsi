"""Spectral response function helpers for hyperspectral sensors."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

__all__: list[str] = []

if find_spec("alchemi_hsi.srf.hytes") is not None:
    _hytes = import_module("alchemi_hsi.srf.hytes")
    hytes_srf_matrix = getattr(_hytes, "hytes_srf_matrix")
    __all__.append("hytes_srf_matrix")

