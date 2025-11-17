"""Synthetic sensor response generator for SRF-randomised training paths."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..types import SRFMatrix
from .resample import project_to_sensor

try:  # NumPy >= 2.0
    from numpy import trapezoid as _integrate
except ImportError:  # pragma: no cover - fallback for NumPy < 2.0
    from numpy import trapz as _integrate  # type: ignore[attr-defined]

ShapeKind = Literal["gaussian", "box", "hamming"]


@dataclass(slots=True)
class SyntheticSensorConfig:
    """Configuration describing the randomised synthetic sensor."""

    highres_axis_nm: np.ndarray
    n_bands: int
    center_jitter_nm: float = 0.0
    fwhm_range_nm: tuple[float, float] = (5.0, 15.0)
    shape: ShapeKind = "gaussian"
    seed: int | None = None

    def axis(self) -> np.ndarray:
        return _validate_grid(self.highres_axis_nm)


@dataclass(slots=True)
class ProjectedSpectrum:
    """Container for spectra projected onto a synthetic sensor."""

    values: np.ndarray
    centers_nm: np.ndarray
    fwhm_nm: np.ndarray
    srf_matrix: np.ndarray
    srf_axis_nm: np.ndarray


@dataclass(frozen=True)
class _RandomSpec:
    centers_nm: np.ndarray
    fwhm_nm: np.ndarray
    dense_resp: np.ndarray


def rand_srf_grid(
    highres_axis: Sequence[float],
    *,
    n_bands: int,
    center_jitter_nm: float,
    fwhm_range_nm: Iterable[float] | float,
    shape: ShapeKind,
    seed: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a dense SRF matrix expressed on ``highres_axis`` (nm)."""

    wl = _validate_grid(highres_axis)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    spec = _generate_random_spec(
        wl,
        n_bands=n_bands,
        center_jitter_nm=center_jitter_nm,
        fwhm_range_nm=fwhm_range_nm,
        shape=shape,
        rng=rng,
    )
    return spec.centers_nm.copy(), spec.dense_resp.copy()


def project_lab_to_synthetic(
    lab_values: Sequence[float],
    lab_axis_nm: Sequence[float],
    cfg: SyntheticSensorConfig,
    *,
    seed: int | None = None,
) -> ProjectedSpectrum:
    """Project a lab spectrum onto a randomly sampled synthetic sensor."""

    lab_axis = _validate_grid(lab_axis_nm)
    lab_vals = np.asarray(lab_values, dtype=np.float64)
    if lab_vals.ndim != 1 or lab_vals.shape[0] != lab_axis.shape[0]:
        raise ValueError("lab_values must align with lab_axis_nm")

    highres = cfg.axis()
    interp = np.interp(highres, lab_axis, lab_vals)

    rng_seed = seed if seed is not None else cfg.seed
    centers, dense = rand_srf_grid(
        highres,
        n_bands=cfg.n_bands,
        center_jitter_nm=cfg.center_jitter_nm,
        fwhm_range_nm=cfg.fwhm_range_nm,
        shape=cfg.shape,
        seed=rng_seed,
    )
    matrix = _dense_to_matrix(highres, centers, dense, cfg.shape)
    projected = project_to_sensor(highres, interp, centers, srf=matrix)

    fwhm = np.asarray([estimate_fwhm(highres, row) for row in dense], dtype=np.float64)
    return ProjectedSpectrum(
        values=np.asarray(projected, dtype=np.float64),
        centers_nm=centers,
        fwhm_nm=fwhm,
        srf_matrix=dense.astype(np.float32, copy=False),
        srf_axis_nm=highres.copy(),
    )


def _dense_to_matrix(
    highres_axis: np.ndarray,
    centers: np.ndarray,
    dense: np.ndarray,
    shape: ShapeKind,
) -> SRFMatrix:
    bands_nm = [highres_axis.copy() for _ in range(dense.shape[0])]
    bands_resp = [row.astype(np.float64, copy=True) for row in dense]
    matrix = SRFMatrix(
        sensor=f"synthetic_{shape}",
        centers_nm=centers.copy(),
        bands_nm=bands_nm,
        bands_resp=bands_resp,
        version="synthetic",
    )
    return matrix.normalize_trapz()


def _validate_grid(grid_nm: Sequence[float]) -> np.ndarray:
    wl = np.asarray(grid_nm, dtype=np.float64)
    if wl.ndim != 1:
        raise ValueError("Wavelength grid must be 1-D")
    if wl.size < 2:
        raise ValueError("Wavelength grid must contain at least two samples")
    if np.any(np.diff(wl) <= 0):
        raise ValueError("Wavelength grid must be strictly increasing")
    return wl


def _validate_fwhm_range(
    fwhm_range_nm: Iterable[float] | float, n_bands: int
) -> tuple[np.ndarray, bool]:
    if np.isscalar(fwhm_range_nm):
        width = float(fwhm_range_nm)
        if not np.isfinite(width) or width <= 0:
            raise ValueError("FWHM must be positive")
        return np.full(n_bands, width, dtype=np.float64), True

    values = np.asarray(tuple(fwhm_range_nm), dtype=np.float64)
    if values.size != 2:
        raise ValueError("fwhm_range_nm must provide exactly two values when iterable")
    lo, hi = float(values[0]), float(values[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0 or hi <= 0:
        raise ValueError("FWHM range bounds must be positive and finite")
    if hi < lo:
        raise ValueError("FWHM range upper bound must be >= lower bound")
    return np.array([lo, hi], dtype=np.float64), False


def _draw_centers(
    wl: np.ndarray,
    n_bands: int,
    center_jitter_nm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if center_jitter_nm < 0:
        raise ValueError("center_jitter_nm must be non-negative")

    base = np.linspace(wl[0], wl[-1], n_bands + 2, dtype=np.float64)[1:-1]
    if center_jitter_nm == 0:
        centers = base
    else:
        jitter = rng.uniform(-center_jitter_nm, center_jitter_nm, size=n_bands)
        centers = base + jitter
        np.clip(centers, wl[0], wl[-1], out=centers)
    return np.asarray(np.sort(centers), dtype=np.float64)


def _band_response(
    wl: np.ndarray,
    center: float,
    fwhm_nm: float,
    shape: ShapeKind,
) -> np.ndarray:
    if fwhm_nm <= 0 or not np.isfinite(fwhm_nm):
        raise ValueError("FWHM must be positive and finite")

    if shape == "gaussian":
        sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        weights = np.exp(-0.5 * ((wl - center) / sigma) ** 2)
    else:
        half = 0.5 * fwhm_nm
        support = np.abs(wl - center) <= half
        if support.sum() < 2:
            idx = int(np.argmin(np.abs(wl - center)))
            support[idx] = True
            if idx == 0 and wl.size > 1:
                support[1] = True
            elif idx == wl.size - 1 and wl.size > 1:
                support[-2] = True
            elif wl.size > 2:
                support[idx - 1] = True
        weights = np.zeros_like(wl)
        if shape == "box":
            weights[support] = 1.0
        elif shape == "hamming":
            if not np.any(support):
                return weights
            span = max(fwhm_nm, np.finfo(np.float64).eps)
            norm = (wl[support] - (center - half)) / span
            weights[support] = 0.54 - 0.46 * np.cos(2.0 * np.pi * norm)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported shape '{shape}'")

    area = float(_integrate(weights, wl))
    if not np.isfinite(area) or area <= 0:
        raise ValueError("SRF band must integrate to a positive finite area")
    return weights / area


def _generate_random_spec(
    wl: np.ndarray,
    *,
    n_bands: int,
    center_jitter_nm: float,
    fwhm_range_nm: Iterable[float] | float,
    shape: ShapeKind,
    rng: np.random.Generator,
) -> _RandomSpec:
    if n_bands <= 0:
        raise ValueError("n_bands must be positive")

    centers = _draw_centers(wl, n_bands, center_jitter_nm, rng)
    fwhm_range, is_scalar = _validate_fwhm_range(fwhm_range_nm, n_bands)
    if is_scalar:
        fwhm = fwhm_range
    else:
        lo, hi = fwhm_range
        if lo == hi:
            fwhm = np.full(n_bands, lo, dtype=np.float64)
        else:
            fwhm = rng.uniform(lo, hi, size=n_bands)

    dense_resp = np.zeros((n_bands, wl.shape[0]), dtype=np.float64)
    for idx, (center, width) in enumerate(zip(centers, fwhm, strict=True)):
        dense_resp[idx] = _band_response(wl, float(center), float(width), shape)

    return _RandomSpec(centers, np.asarray(fwhm, dtype=np.float64), dense_resp)


def estimate_fwhm(band_wl_nm: np.ndarray, band_resp: np.ndarray) -> float:
    """Estimate the FWHM of a normalized SRF row."""

    wl = np.asarray(band_wl_nm, dtype=np.float64)
    resp = np.asarray(band_resp, dtype=np.float64)
    if wl.ndim != 1 or resp.ndim != 1 or wl.shape != resp.shape:
        raise ValueError("Band wavelength/response arrays must be 1-D and matched")
    if resp.size < 2:
        return 0.0

    peak = float(resp.max())
    if peak <= 0.0 or not np.isfinite(peak):
        return 0.0
    half = 0.5 * peak
    above = resp >= half
    if not np.any(above):
        return 0.0

    idx = np.flatnonzero(above)
    left = idx[0]
    right = idx[-1]

    if left == right:
        return 0.0

    if left == 0:
        left_pos = wl[left]
    else:
        x0, x1 = wl[left - 1], wl[left]
        y0, y1 = resp[left - 1], resp[left]
        frac_left = (half - y0) / (y1 - y0) if y1 != y0 else 0.0
        left_pos = x0 + frac_left * (x1 - x0)

    if right == wl.size - 1:
        right_pos = wl[right]
    else:
        x0r, x1r = wl[right], wl[right + 1]
        y0r, y1r = resp[right], resp[right + 1]
        if x1r == x0r:
            right_pos = x0r
        else:
            frac_right = (half - y0r) / (y1r - y0r) if y1r != y0r else 0.0
            right_pos = x0r + frac_right * (x1r - x0r)

    return float(right_pos - left_pos)


__all__ = [
    "ProjectedSpectrum",
    "SyntheticSensorConfig",
    "estimate_fwhm",
    "project_lab_to_synthetic",
    "rand_srf_grid",
]

