"""Utility helpers for SRF-aware tokenisation paths."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..types import SRFMatrix
from ..utils.integrate import np_integrate as _np_integrate
from .synthetic import estimate_fwhm

if TYPE_CHECKING:
    from .registry import SRFRegistry

_KNOWN_SENSORS = {"emit", "enmap", "avirisng", "hytes"}
_DEFAULT_SRF_DIR = Path("data") / "srf"


def _ensure_cache_dir() -> Path:
    path = _DEFAULT_SRF_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_sensor_srf(
    sensor_id: str | None, *, registry: "SRFRegistry" | None = None
) -> SRFMatrix | None:
    """Best-effort retrieval of an SRF matrix for ``sensor_id``."""

    from .registry import SRFRegistry, get_srf

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


def _coerce_srf_inputs(
    wavelength_nm: np.ndarray, srfs: np.ndarray
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    wl = np.asarray(wavelength_nm, dtype=np.float64)
    responses = np.asarray(srfs, dtype=np.float64)
    if wl.ndim != 1:
        raise ValueError("wavelength_nm must be one-dimensional")
    if responses.ndim != 2:
        raise ValueError("srfs must be two-dimensional with shape [bands, wavelengths]")
    if responses.shape[1] != wl.shape[0]:
        raise ValueError("srfs column count must match wavelength grid length")
    return wl, responses


def normalize_srf_rows(
    wavelength_nm: np.ndarray, srfs: np.ndarray, *, atol: float = 1e-3
) -> NDArray[np.float64]:
    """Normalize SRF rows with trapezoidal integration over ``wavelength_nm``.

    Parameters
    ----------
    wavelength_nm:
        Shared wavelength grid for the SRF samples (shape ``[L_srf]``).
    srfs:
        Response matrix shaped ``[bands, L_srf]``.
    atol:
        Minimum absolute area allowed before normalization. Rows whose area falls
        below this threshold are treated as invalid and raise a ``ValueError``.
    """

    wl, responses = _coerce_srf_inputs(wavelength_nm, srfs)
    areas = _np_integrate(responses, wl, axis=1)
    normalized = np.asarray(responses, dtype=np.float64).copy()

    for idx, area in enumerate(areas):
        if not np.isfinite(area):
            raise ValueError(f"SRF row {idx} integrates to a non-finite area ({area})")
        if abs(area) <= atol:
            raise ValueError(f"SRF row {idx} has near-zero area ({area}); cannot normalize")
        normalized[idx] = responses[idx] / area

    return normalized


def validate_srf(
    wavelength_nm: np.ndarray,
    srfs: np.ndarray,
    *,
    area_tol: float = 1e-3,
    allow_negative_eps: float = 0.0,
    min_area: float = 1e-6,
) -> None:
    """Validate SRF normalization, support, and sign constraints.

    Raises a ``ValueError`` with a descriptive message when any band violates the
    specified tolerance bounds.
    """

    wl, responses = _coerce_srf_inputs(wavelength_nm, srfs)
    areas = _np_integrate(responses, wl, axis=1)

    for idx, (row, area) in enumerate(zip(responses, areas, strict=True)):
        if not np.isfinite(area):
            raise ValueError(f"SRF row {idx} integrates to a non-finite area ({area})")
        if area < min_area:
            raise ValueError(f"SRF row {idx} area {area:.3e} is below minimum {min_area}")
        if abs(area - 1.0) > area_tol:
            raise ValueError(
                f"SRF row {idx} area {area:.3f} deviates from 1.0 beyond tolerance {area_tol}"
            )
        min_val = float(np.min(row))
        if min_val < -allow_negative_eps:
            raise ValueError(
                f"SRF row {idx} contains negative entries (min={min_val}) below allowed"
                f" epsilon {allow_negative_eps}"
            )


def check_flat_spectrum_invariant(
    wavelength_nm: np.ndarray,
    srfs: np.ndarray,
    *,
    c: float = 1.0,
    tol: float = 1e-3,
) -> None:
    """Assert that a flat spectrum remains flat after SRF convolution."""

    wl, responses = _coerce_srf_inputs(wavelength_nm, srfs)
    spectrum = np.full_like(wl, fill_value=float(c), dtype=np.float64)
    areas = _np_integrate(responses, wl, axis=1)
    if np.any(areas <= 0.0):
        raise AssertionError("SRF areas must be positive before convolution check")
    convolved = _np_integrate(responses * spectrum[None, :], wl, axis=1) / areas
    if not np.allclose(convolved, c, atol=tol):
        raise AssertionError(
            "Flat spectrum invariant violated: convolved values differ from input constant"
        )


def default_band_widths(
    sensor_id: str | None,
    axis_nm: np.ndarray,
    *,
    registry: "SRFRegistry" | None = None,
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
    "check_flat_spectrum_invariant",
    "load_sensor_srf",
    "normalize_srf_rows",
    "validate_srf",
]
