"""Synthetic sensor response generator for SRF-randomised training paths."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..spectral.srf import SRFMatrix as DenseSRFMatrix
from ..types import QuantityKind, SRFMatrix, Spectrum, ValueUnits
from ..utils.integrate import np_integrate as _np_integrate
from .registry import SensorSRF, SRFProvenance
from .resample import project_to_sensor

ShapeKind = Literal["gaussian", "box", "hamming"]


@dataclass(slots=True)
class SyntheticSensorConfig:
    """Configuration describing the randomised synthetic sensor."""

    highres_axis_nm: NDArray[np.float64]
    n_bands: int
    center_jitter_nm: float = 0.0
    fwhm_range_nm: tuple[float, float] = (5.0, 15.0)
    shape: ShapeKind = "gaussian"
    seed: int | None = None

    def axis(self) -> NDArray[np.float64]:
        return _validate_grid(self.highres_axis_nm)


@dataclass(slots=True)
class ProjectedSpectrum:
    """Container for spectra projected onto a synthetic sensor."""

    values: NDArray[np.float64]
    centers_nm: NDArray[np.float64]
    fwhm_nm: NDArray[np.float64]
    srf_matrix: NDArray[np.float64]
    srf_axis_nm: NDArray[np.float64]


@dataclass(slots=True)
class SRFJitterConfig:
    """Configuration controlling real-sensor SRF jitter."""

    enabled: bool = False
    center_shift_std_nm: float = 0.0
    width_scale_std: float = 0.0
    shape_jitter_std: float = 0.0
    seed: int | None = None
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        if self.center_shift_std_nm < 0:
            raise ValueError("center_shift_std_nm must be non-negative")
        if self.width_scale_std < 0:
            raise ValueError("width_scale_std must be non-negative")
        if self.shape_jitter_std < 0:
            raise ValueError("shape_jitter_std must be non-negative")

    def generator(self, fallback: np.random.Generator | None = None) -> np.random.Generator:
        if fallback is not None:
            return fallback
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)
        return self.rng


@dataclass(frozen=True)
class _RandomSpec:
    centers_nm: NDArray[np.float64]
    fwhm_nm: NDArray[np.float64]
    dense_resp: NDArray[np.float64]


def make_gaussian_srf(
    centers_nm: np.ndarray,
    fwhm_nm: np.ndarray,
    *,
    wavelength_grid_nm: np.ndarray,
) -> np.ndarray:
    """Construct a dense Gaussian SRF matrix from band metadata."""

    centers = np.asarray(centers_nm, dtype=np.float64)
    widths = np.asarray(fwhm_nm, dtype=np.float64)
    wl = np.asarray(wavelength_grid_nm, dtype=np.float64)

    if centers.ndim != 1 or widths.ndim != 1:
        raise ValueError("centers_nm and fwhm_nm must be 1-D arrays")
    if centers.shape[0] != widths.shape[0]:
        raise ValueError("centers_nm and fwhm_nm must have matching lengths")
    if wl.ndim != 1:
        raise ValueError("wavelength_grid_nm must be 1-D")
    if centers.size == 0:
        return np.empty((0, wl.shape[0]), dtype=np.float64)

    sigma = widths / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma = sigma[:, None]
    shifted = wl[None, :] - centers[:, None]
    return np.exp(-0.5 * (shifted / sigma) ** 2)


def rand_srf_grid(
    highres_axis: ArrayLike,
    *,
    n_bands: int,
    center_jitter_nm: float,
    fwhm_range_nm: Iterable[float] | float,
    shape: ShapeKind,
    seed: int | np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
    highres_axis: NDArray[np.float64],
    centers: NDArray[np.float64],
    dense: NDArray[np.float64],
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


def _validate_grid(grid_nm: ArrayLike) -> NDArray[np.float64]:
    wl = np.asarray(grid_nm, dtype=np.float64)
    if wl.ndim != 1:
        raise ValueError("Wavelength grid must be 1-D")
    if wl.size < 2:
        raise ValueError("Wavelength grid must contain at least two samples")
    if np.any(np.diff(wl) <= 0):
        raise ValueError("Wavelength grid must be strictly increasing")
    return np.asarray(wl, dtype=np.float64)


def _validate_fwhm_range(
    fwhm_range_nm: Iterable[float] | float, n_bands: int
) -> tuple[NDArray[np.float64], bool]:
    if isinstance(fwhm_range_nm, (int, float)):
        width = float(np.asarray(fwhm_range_nm, dtype=np.float64))
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
    wl: NDArray[np.float64],
    n_bands: int,
    center_jitter_nm: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
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
    wl: NDArray[np.float64],
    center: float,
    fwhm_nm: float,
    shape: ShapeKind,
) -> NDArray[np.float64]:
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

    area = float(_np_integrate(weights, wl))
    if not np.isfinite(area) or area <= 0:
        raise ValueError("SRF band must integrate to a positive finite area")
    return np.asarray(weights / area, dtype=np.float64)


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


def _normalize_and_validate_srf(
    wavelength_grid_nm: np.ndarray, srfs: np.ndarray, *, tol: float = 1e-6
) -> np.ndarray:
    wl = np.asarray(wavelength_grid_nm, dtype=np.float64)
    matrix = DenseSRFMatrix(wl, np.maximum(srfs, 0.0))
    matrix.assert_nonnegative(tol=0.0)
    matrix.normalize_rows_trapz()

    flat = Spectrum(
        wavelength_nm=wl,
        values=np.ones(wl.shape[0], dtype=np.float64),
        kind=QuantityKind.SURFACE_REFLECTANCE,
        units=ValueUnits.REFLECTANCE_FRACTION,
    )
    matrix.assert_flat_spectrum_preserved(flat, tol=tol)
    return np.asarray(matrix.matrix, dtype=np.float64)


def make_virtual_sensor(
    *,
    sensor_id: str = "virtual",
    wavelength_min_nm: float,
    wavelength_max_nm: float,
    band_count: int,
    base_fwhm_nm: float | None = None,
    center_jitter_nm: float = 0.0,
    width_jitter_frac: float = 0.0,
    grid_step_nm: float = 1.0,
    rng: np.random.Generator | None = None,
) -> SensorSRF:
    rng = rng or np.random.default_rng()
    if band_count <= 0:
        raise ValueError("band_count must be positive")
    if wavelength_max_nm <= wavelength_min_nm:
        raise ValueError("wavelength_max_nm must exceed wavelength_min_nm")
    if grid_step_nm <= 0:
        raise ValueError("grid_step_nm must be positive")
    if center_jitter_nm < 0:
        raise ValueError("center_jitter_nm must be non-negative")
    if width_jitter_frac < 0:
        raise ValueError("width_jitter_frac must be non-negative")

    wavelength_grid_nm = np.arange(
        wavelength_min_nm,
        wavelength_max_nm + grid_step_nm / 2.0,
        grid_step_nm,
        dtype=np.float64,
    )

    centers = rng.uniform(wavelength_min_nm, wavelength_max_nm, size=band_count)
    centers.sort()

    span = wavelength_max_nm - wavelength_min_nm
    default_width = span / max(band_count * 1.5, 1.0)
    widths = np.full(band_count, base_fwhm_nm or default_width, dtype=np.float64)

    if center_jitter_nm > 0:
        centers += rng.uniform(-center_jitter_nm, center_jitter_nm, size=centers.shape)
        np.clip(centers, wavelength_min_nm, wavelength_max_nm, out=centers)

    if width_jitter_frac > 0:
        jitter = rng.uniform(1.0 - width_jitter_frac, 1.0 + width_jitter_frac, size=widths.shape)
        widths *= jitter

    widths = np.clip(widths, np.finfo(np.float64).eps, None)

    raw_srfs = make_gaussian_srf(centers, widths, wavelength_grid_nm=wavelength_grid_nm)
    normalized = _normalize_and_validate_srf(wavelength_grid_nm, raw_srfs)

    meta = {"virtual": True, "grid_step_nm": grid_step_nm}
    return SensorSRF(
        sensor_id=sensor_id,
        wavelength_grid_nm=wavelength_grid_nm,
        srfs=normalized,
        band_centers_nm=centers,
        band_widths_nm=widths,
        provenance=SRFProvenance.SYNTHETIC,
        meta=meta,
    )


def perturb_sensor_srf(
    sensor_srf: SensorSRF,
    *,
    center_jitter_nm: float = 0.0,
    width_jitter_frac: float = 0.0,
    shape_noise_frac: float = 0.0,
    rng: np.random.Generator | None = None,
) -> SensorSRF:
    rng = rng or np.random.default_rng()

    if center_jitter_nm < 0:
        raise ValueError("center_jitter_nm must be non-negative")
    if width_jitter_frac < 0:
        raise ValueError("width_jitter_frac must be non-negative")
    if shape_noise_frac < 0:
        raise ValueError("shape_noise_frac must be non-negative")

    base_centers = np.asarray(sensor_srf.band_centers_nm, dtype=np.float64)
    base_widths = np.asarray(sensor_srf.band_widths_nm, dtype=np.float64)
    centers = base_centers.copy()
    widths = base_widths.copy()
    srfs = np.asarray(sensor_srf.srfs, dtype=np.float64).copy()

    if center_jitter_nm > 0:
        centers += rng.uniform(-center_jitter_nm, center_jitter_nm, size=centers.shape)
        np.clip(
            centers,
            sensor_srf.wavelength_grid_nm[0],
            sensor_srf.wavelength_grid_nm[-1],
            out=centers,
        )

    if width_jitter_frac > 0:
        widths *= rng.uniform(1.0 - width_jitter_frac, 1.0 + width_jitter_frac, size=widths.shape)
        widths = np.clip(widths, np.finfo(np.float64).eps, None)

    if shape_noise_frac > 0:
        noise = rng.normal(scale=shape_noise_frac, size=srfs.shape) * srfs
        srfs = np.clip(srfs + noise, 0.0, None)

    normalized = _normalize_and_validate_srf(sensor_srf.wavelength_grid_nm, srfs)

    meta = dict(sensor_srf.meta)
    meta["perturbed"] = True

    return SensorSRF(
        sensor_id=f"{sensor_srf.sensor_id}_perturbed",
        wavelength_grid_nm=sensor_srf.wavelength_grid_nm.copy(),
        srfs=normalized,
        band_centers_nm=centers,
        band_widths_nm=widths,
        provenance=sensor_srf.provenance,
        valid_mask=sensor_srf.valid_mask,
        meta=meta,
    )


def jitter_sensor_srf(
    sensor_srf: SensorSRF,
    jitter_cfg: SRFJitterConfig,
    *,
    rng: np.random.Generator | None = None,
) -> SensorSRF:
    """Apply Gaussian-distributed jitter to a sensor SRF copy."""

    if not jitter_cfg.enabled:
        return sensor_srf

    rng = jitter_cfg.generator(rng)

    base_centers = np.asarray(sensor_srf.band_centers_nm, dtype=np.float64)
    base_widths = np.asarray(sensor_srf.band_widths_nm, dtype=np.float64)
    centers = base_centers.copy()
    widths = base_widths.copy()
    srfs = np.asarray(sensor_srf.srfs, dtype=np.float64).copy()

    if jitter_cfg.center_shift_std_nm > 0:
        centers += rng.normal(scale=jitter_cfg.center_shift_std_nm, size=centers.shape)
        np.clip(
            centers,
            sensor_srf.wavelength_grid_nm[0],
            sensor_srf.wavelength_grid_nm[-1],
            out=centers,
        )
        centers = np.maximum.accumulate(centers)

        positive_diffs = np.diff(np.asarray(sensor_srf.band_centers_nm, dtype=np.float64))
        min_spacing = float(np.min(positive_diffs[positive_diffs > 0])) if positive_diffs.size else 0.0
        min_step = max(min_spacing * 0.2, np.finfo(np.float64).eps)

        for idx in range(1, centers.size):
            if centers[idx] <= centers[idx - 1]:
                centers[idx] = min(
                    centers[idx - 1] + min_step,
                    sensor_srf.wavelength_grid_nm[-1],
                )

    if jitter_cfg.width_scale_std > 0:
        widths *= np.exp(rng.normal(scale=jitter_cfg.width_scale_std, size=widths.shape))
        widths = np.clip(widths, np.finfo(np.float64).eps, None)

    width_scale = widths / base_widths

    if centers.shape[0] == srfs.shape[0]:
        for idx, (row, delta, scale, base_center) in enumerate(
            zip(srfs, centers - base_centers, width_scale, base_centers, strict=True)
        ):
            if delta == 0.0 and scale == 1.0:
                continue
            source_axis = (sensor_srf.wavelength_grid_nm - delta - base_center) / scale + base_center
            srfs[idx] = np.interp(
                sensor_srf.wavelength_grid_nm,
                source_axis,
                row,
                left=0.0,
                right=0.0,
            )

    if jitter_cfg.shape_jitter_std > 0:
        noise = rng.normal(scale=jitter_cfg.shape_jitter_std, size=srfs.shape)
        srfs = np.clip(srfs * (1.0 + noise), 0.0, None)

    normalized = _normalize_and_validate_srf(sensor_srf.wavelength_grid_nm, srfs)

    meta = dict(sensor_srf.meta)
    meta["jitter"] = {
        "center_shift_std_nm": jitter_cfg.center_shift_std_nm,
        "width_scale_std": jitter_cfg.width_scale_std,
        "shape_jitter_std": jitter_cfg.shape_jitter_std,
        "seed": jitter_cfg.seed,
    }

    return SensorSRF(
        sensor_id=sensor_srf.sensor_id,
        wavelength_grid_nm=sensor_srf.wavelength_grid_nm.copy(),
        srfs=normalized,
        band_centers_nm=centers,
        band_widths_nm=widths,
        provenance=sensor_srf.provenance,
        valid_mask=sensor_srf.valid_mask,
        meta=meta,
    )


__all__ = [
    "ProjectedSpectrum",
    "SRFJitterConfig",
    "SyntheticSensorConfig",
    "estimate_fwhm",
    "make_gaussian_srf",
    "jitter_sensor_srf",
    "make_virtual_sensor",
    "perturb_sensor_srf",
    "project_lab_to_synthetic",
    "rand_srf_grid",
]
