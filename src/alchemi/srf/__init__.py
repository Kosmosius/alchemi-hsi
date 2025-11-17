"""Spectral response function utilities with lazy attribute loading."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "SRFRegistry",
    "avirisng_bad_band_mask",
    "avirisng_srf_matrix",
    "batch_convolve_lab_to_sensor",
    "boxcar_resample",
    "build_matrix_from_centers",
    "convolve_lab_to_sensor",
    "convolve_to_bands",
    "emit_srf_matrix",
    "enmap_srf_matrix",
    "gaussian_resample",
    "gaussian_srf",
    "hytes_srf_matrix",
    "MAKO_BAND_COUNT",
    "build_mako_srf_from_header",
    "get_srf",
    "mako_lwir_grid_nm",
    "project_to_sensor",
    "rand_srf_grid",
    "validate_srf_matrix",
]

_EXPORTS = {
    "SRFRegistry": ("alchemi.srf.registry", "SRFRegistry"),
    "avirisng_bad_band_mask": ("alchemi.srf.avirisng", "avirisng_bad_band_mask"),
    "avirisng_srf_matrix": ("alchemi.srf.avirisng", "avirisng_srf_matrix"),
    "batch_convolve_lab_to_sensor": ("alchemi.srf.batch_convolve", "batch_convolve_lab_to_sensor"),
    "boxcar_resample": ("alchemi.srf.resample", "boxcar_resample"),
    "build_matrix_from_centers": ("alchemi.srf.fallback", "build_matrix_from_centers"),
    "convolve_lab_to_sensor": ("alchemi.srf.convolve", "convolve_lab_to_sensor"),
    "convolve_to_bands": ("alchemi.srf.resample", "convolve_to_bands"),
    "emit_srf_matrix": ("alchemi.srf.emit", "emit_srf_matrix"),
    "enmap_srf_matrix": ("alchemi.srf.enmap", "enmap_srf_matrix"),
    "gaussian_resample": ("alchemi.srf.resample", "gaussian_resample"),
    "gaussian_srf": ("alchemi.srf.fallback", "gaussian_srf"),
    "hytes_srf_matrix": ("alchemi.srf.hytes", "hytes_srf_matrix"),
    "MAKO_BAND_COUNT": ("alchemi.srf.mako", "MAKO_BAND_COUNT"),
    "build_mako_srf_from_header": ("alchemi.srf.mako", "build_mako_srf_from_header"),
    "get_srf": ("alchemi.srf.registry", "get_srf"),
    "mako_lwir_grid_nm": ("alchemi.srf.mako", "mako_lwir_grid_nm"),
    "project_to_sensor": ("alchemi.srf.resample", "project_to_sensor"),
    "rand_srf_grid": ("alchemi.srf.synthetic", "rand_srf_grid"),
    "validate_srf_matrix": ("alchemi.srf.fallback", "validate_srf_matrix"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = importlib.import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value
