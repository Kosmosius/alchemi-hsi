"""Utility helpers for SRF-aware tokenisation paths."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..types import SRFMatrix
from ..utils.integrate import np_integrate as _np_integrate
from .registry import SRFRegistry, get_srf
from .synthetic import estimate_fwhm

_KNOWN_SENSORS = {"emit", "enmap", "avirisng", "hytes"}
_DEFAULT_SRF_DIR = Path("data") / "srf"


def _ensure_cache_dir() -> Path:
    path = _DEFAULT_SRF_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_sensor_srf(
    sensor_id: str | None, *, registry: SRFRegistry | None = None
) -> SRFMatrix | None:
    """Best-effort retrieval of an SRF matrix for ``sensor_id``."""

    if not sensor_id:
        return None
    sensor = sensor_id.lower()
    if registry is not None:
        try:
            return registry.get(sensor)
        except FileNotFoundError:
            pass

    if sensor == "emit":
        srf, _ = get_srf("emit")
        return srf
    if sensor == "enmap":
        from .enmap import enmap_srf_matrix

        return enmap_srf_matrix(cache_dir=_ensure_cache_dir())
    if sensor == "avirisng":
        from .avirisng import avirisng_srf_matrix

        return avirisng_srf_matrix(cache_dir=_ensure_cache_dir())
    if sensor == "hytes":
        from .hytes import hytes_srf_matrix

        return hytes_srf_matrix()
    return None


def default_band_widths(
    sensor_id: str | None,
    axis_nm: np.ndarray,
    *,
    registry: SRFRegistry | None = None,
    srf: SRFMatrix | None = None,
) -> NDArray[np.float64]:
    """Return estimated FWHM values aligned with ``axis_nm``."""

    wavelengths = np.asarray(axis_nm, dtype=np.float64)
    if wavelengths.ndim != 1:
        raise ValueError("axis_nm must be one-dimensional")

    sensor = sensor_id.lower() if sensor_id else None
    if srf is None and sensor in _KNOWN_SENSORS:
        srf = load_sensor_srf(sensor, registry=registry)

    if srf is not None and np.allclose(srf.centers_nm, wavelengths, atol=1e-6):
        widths = np.asarray(
            [
                estimate_fwhm(nm, resp)
                for nm, resp in zip(srf.bands_nm, srf.bands_resp, strict=True)
            ],
            dtype=np.float64,
        )
        if np.any(np.isfinite(widths)):
            fallback = float(np.nanmean(widths[np.isfinite(widths)]))
            widths = np.where(np.isfinite(widths), widths, fallback)
            return np.asarray(widths, dtype=np.float64)

    if sensor == "emit":
        widths = np.full(wavelengths.shape, 7.25, dtype=np.float64)
    elif sensor == "enmap":
        widths = np.where(wavelengths < 1_000.0, 8.5, 12.0)
    elif sensor == "avirisng":
        widths = np.where(wavelengths < 1_000.0, 6.0, np.where(wavelengths < 1_800.0, 7.5, 9.0))
    elif sensor == "hytes":
        diffs = np.diff(wavelengths)
        width = float(np.median(diffs)) if diffs.size else 44.0
        widths = np.full(wavelengths.shape, max(width, 1.0), dtype=np.float64)
    else:
        widths = _band_spacing(wavelengths)

    widths[widths <= 0] = float(np.mean(widths[widths > 0])) if np.any(widths > 0) else 1.0
    return np.asarray(widths, dtype=np.float64)


def build_srf_band_embeddings(
    srf: SRFMatrix,
    *,
    summary_stats: Iterable[str] | None = None,
) -> NDArray[np.float32]:
    """Compress SRF rows into a compact embedding vector."""

    stats = tuple(summary_stats or ("mean", "std", "skew", "kurt", "peak", "area"))
    features: list[list[float]] = []

    for nm, resp in zip(srf.bands_nm, srf.bands_resp, strict=True):
        wl = np.asarray(nm, dtype=np.float64)
        weights = np.asarray(resp, dtype=np.float64)
        area = float(_np_integrate(weights, wl))
        if area <= 0 or not np.isfinite(area):
            weights = np.ones_like(wl) / wl.size
            area = 1.0
        pdf = weights / area
        mean = float(_np_integrate(wl * pdf, wl))
        centered = wl - mean
        variance = float(_np_integrate((centered**2) * pdf, wl))
        std = float(np.sqrt(max(variance, 1e-12)))
        skew = float(_np_integrate((centered**3) * pdf, wl) / (std**3 + 1e-12))
        kurt = float(_np_integrate((centered**4) * pdf, wl) / (std**4 + 1e-12))
        peak = float(np.max(pdf))
        row: list[float] = []
        for key in stats:
            match key:
                case "mean":
                    row.append(mean)
                case "std":
                    row.append(std)
                case "skew":
                    row.append(skew)
                case "kurt":
                    row.append(kurt)
                case "peak":
                    row.append(peak)
                case "area":
                    row.append(area)
                case _:
                    raise ValueError(f"Unsupported SRF stat '{key}'")
        features.append(row)

    return np.asarray(features, dtype=np.float32)


def _band_spacing(axis_nm: np.ndarray) -> NDArray[np.float64]:
    if axis_nm.size == 0:
        return np.empty(0, dtype=np.float64)
    diffs = np.diff(axis_nm)
    if diffs.size == 0:
        return np.full(axis_nm.shape, 1.0, dtype=np.float64)
    widths = np.empty_like(axis_nm)
    widths[0] = abs(diffs[0])
    widths[-1] = abs(diffs[-1])
    widths[1:-1] = 0.5 * (np.abs(diffs[:-1]) + np.abs(diffs[1:]))
    return np.asarray(widths, dtype=np.float64)


__all__ = [
    "build_srf_band_embeddings",
    "default_band_widths",
    "load_sensor_srf",
]
