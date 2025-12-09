"""Spectral response function utilities with lazy attribute loading."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "MAKO_BAND_COUNT",
    "ProjectedSpectrum",
    "SensorSRF",
    "SRFProvenance",
    "SRFRegistry",
    "SensorSRFRegistry",
    "GLOBAL_SRF_REGISTRY",
    "normalize_srf_rows",
    "SensorSRF",
    "SRFProvenance",
    "SyntheticSensorConfig",
    "avirisng_bad_band_mask",
    "avirisng_srf_matrix",
    "batch_convolve_lab_to_sensor",
    "boxcar_resample",
    "resample_with_srf",
    "resample_values_with_srf",
    "resample_to_sensor",
    "build_mako_srf_from_header",
    "build_matrix_from_centers",
    "make_gaussian_srf",
    "make_virtual_sensor",
    "perturb_sensor_srf",
    "convolve_lab_to_sensor",
    "convolve_to_bands",
    "emit_srf_matrix",
    "enmap_srf_matrix",
    "gaussian_resample",
    "gaussian_srf",
    "get_srf",
    "hytes_srf_matrix",
    "mako_lwir_grid_nm",
    "register_virtual_sensor",
    "project_lab_to_synthetic",
    "project_to_sensor",
    "rand_srf_grid",
    "validate_srf_matrix",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "SRFRegistry": ("alchemi.srf.registry", "SRFRegistry"),
    "SensorSRFRegistry": ("alchemi.srf.registry", "SensorSRFRegistry"),
    "GLOBAL_SRF_REGISTRY": ("alchemi.srf.registry", "GLOBAL_SRF_REGISTRY"),
    "SensorSRF": ("alchemi.spectral.srf", "SensorSRF"),
    "SRFProvenance": ("alchemi.spectral.srf", "SRFProvenance"),
    "get_srf": ("alchemi.srf.registry", "get_srf"),
    "normalize_srf_rows": ("alchemi.srf.sensor", "normalize_srf_rows"),
    "avirisng_bad_band_mask": ("alchemi.srf.avirisng", "avirisng_bad_band_mask"),
    "avirisng_srf_matrix": ("alchemi.srf.avirisng", "avirisng_srf_matrix"),
    "batch_convolve_lab_to_sensor": (
        "alchemi.srf.batch_convolve",
        "batch_convolve_lab_to_sensor",
    ),
    "boxcar_resample": ("alchemi.srf.resample", "boxcar_resample"),
    "resample_with_srf": ("alchemi.srf.resample", "resample_with_srf"),
    "resample_values_with_srf": ("alchemi.srf.resample", "resample_values_with_srf"),
    "resample_to_sensor": ("alchemi.srf.resample", "resample_to_sensor"),
    "convolve_lab_to_sensor": ("alchemi.srf.convolve", "convolve_lab_to_sensor"),
    "convolve_to_bands": ("alchemi.srf.resample", "convolve_to_bands"),
    "emit_srf_matrix": ("alchemi.srf.emit", "emit_srf_matrix"),
    "enmap_srf_matrix": ("alchemi.srf.enmap", "enmap_srf_matrix"),
    "gaussian_resample": ("alchemi.srf.resample", "gaussian_resample"),
    "gaussian_srf": ("alchemi.srf.fallback", "gaussian_srf"),
    "hytes_srf_matrix": ("alchemi.srf.hytes", "hytes_srf_matrix"),
    "MAKO_BAND_COUNT": ("alchemi.srf.mako", "MAKO_BAND_COUNT"),
    "build_mako_srf_from_header": ("alchemi.srf.mako", "build_mako_srf_from_header"),
    "mako_lwir_grid_nm": ("alchemi.srf.mako", "mako_lwir_grid_nm"),
    "build_matrix_from_centers": ("alchemi.srf.fallback", "build_matrix_from_centers"),
    "validate_srf_matrix": ("alchemi.srf.fallback", "validate_srf_matrix"),
    "register_virtual_sensor": ("alchemi.srf.registry", "register_virtual_sensor"),
    "project_to_sensor": ("alchemi.srf.resample", "project_to_sensor"),
    "rand_srf_grid": ("alchemi.srf.synthetic", "rand_srf_grid"),
    "ProjectedSpectrum": ("alchemi.srf.synthetic", "ProjectedSpectrum"),
    "SyntheticSensorConfig": ("alchemi.srf.synthetic", "SyntheticSensorConfig"),
    "make_gaussian_srf": ("alchemi.srf.synthetic", "make_gaussian_srf"),
    "make_virtual_sensor": ("alchemi.srf.synthetic", "make_virtual_sensor"),
    "perturb_sensor_srf": ("alchemi.srf.synthetic", "perturb_sensor_srf"),
    "project_lab_to_synthetic": (
        "alchemi.srf.synthetic",
        "project_lab_to_synthetic",
    ),
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


def __dir__() -> list[str]:
    # So dir(alchemi.srf) shows the lazily exported names
    return sorted(list(globals().keys()) + list(__all__))
