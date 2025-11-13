"""Sensor response function utilities for packaged hyperspectral sensors."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from .avirisng import avirisng_bad_band_mask, avirisng_srf_matrix
from .enmap import enmap_srf_matrix
from .fallback import build_matrix_from_centers, gaussian_srf, validate_srf_matrix

__all__: list[str] = [
  "avirisng_srf_matrix", 
  "avirisng_bad_band_mask", 
  "enmap_srf_matrix",
  "gaussian_srf",
  "build_matrix_from_centers",
  "validate_srf_matrix",
]

if find_spec("alchemi_hsi.srf.hytes") is not None:
    _hytes = import_module("alchemi_hsi.srf.hytes")
    hytes_srf_matrix = getattr(_hytes, "hytes_srf_matrix")
    __all__.append("hytes_srf_matrix")