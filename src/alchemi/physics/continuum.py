"""Continuum removal and absorption feature metrics."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from alchemi.types import QuantityKind, Spectrum, WavelengthGrid

__all__ = [
    "BandMetrics",
    "build_continuum",
    "compute_anchor_continuum",
    "compute_band_depth",
    "compute_band_metrics",
    "compute_convex_hull_continuum",
    "continuum_remove",
    "smooth_continuum",
]


_MIN_DENOM = 1e-12


def _upper_hull_indices(wavelengths: np.ndarray, values: np.ndarray) -> list[int]:
    """Compute indices forming the upper convex hull of (wavelength, value).

    A monotone chain (Andrew's algorithm) is used with wavelength sorted in
    ascending order. Only the upper hull is retained, which is appropriate for
    reflectance continua that should sit above the spectrum.
    """

    points = list(zip(wavelengths, values, strict=False))
    hull: list[int] = []
    for idx, point in enumerate(points):
        while len(hull) >= 2:
            x1, y1 = points[hull[-2]]
            x2, y2 = points[hull[-1]]
            x3, y3 = point
            cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(idx)

    return hull


def _natural_cubic_spline_coefficients(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute natural cubic spline coefficients (b, c, d) for knots (x, y)."""

    n = x.size
    h = np.diff(x)
    alpha = np.zeros(n)
    alpha[1:-1] = (3 / h[1:]) * (y[2:] - y[1:-1]) - (3 / h[:-1]) * (y[1:-1] - y[:-2])

    lower = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)

    for i in range(1, n - 1):
        lower[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / lower[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / lower[i]

    b = np.zeros(n - 1)
    c = np.zeros(n)
    d = np.zeros(n - 1)

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    return b, c[:-1], d


def _evaluate_natural_cubic_spline(
    x: np.ndarray, y: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, x_new: np.ndarray
) -> np.ndarray:
    idx = np.searchsorted(x, x_new, side="right") - 1
    idx = np.clip(idx, 0, x.size - 2)
    dx = x_new - x[idx]
    return y[idx] + b[idx] * dx + c[idx] * dx**2 + d[idx] * dx**3


def _compute_convex_hull_continuum(
    wl: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    hull_idx = _upper_hull_indices(wl, values)
    return wl[hull_idx], values[hull_idx]


def _validate_reflectance_spectrum(spectrum: Spectrum) -> tuple[np.ndarray, np.ndarray]:
    if spectrum.kind not in {
        QuantityKind.REFLECTANCE,
        QuantityKind.SURFACE_REFLECTANCE,
        QuantityKind.TOA_REFLECTANCE,
    }:
        raise ValueError("Continuum removal expects a reflectance-like spectrum")

    wl = np.asarray(spectrum.wavelengths.nm, dtype=np.float64)
    vals = np.asarray(spectrum.values, dtype=np.float64)

    if wl.ndim != 1:
        msg = "Spectrum must use a one-dimensional wavelength grid"
        raise ValueError(msg)
    if vals.shape[-1] != wl.size:
        msg = "Last dimension of values must match wavelength grid length"
        raise ValueError(msg)

    return wl, vals


def _interpolate_continuum(
    wl: np.ndarray, anchor_wl: np.ndarray, anchor_vals: np.ndarray
) -> np.ndarray:
    return np.interp(wl, anchor_wl, anchor_vals)


def compute_convex_hull_continuum(spectrum: Spectrum) -> Spectrum:
    """Compute an upper-hull continuum for a reflectance spectrum.

    The continuum is formed by selecting the upper convex hull of the
    (λ, R(λ)) points and linearly interpolating between those hull vertices.
    This is a robust default for unknown spectra (lab or overhead) because it
    requires no prior knowledge of absorption locations.
    """

    wl, vals = _validate_reflectance_spectrum(spectrum)
    flat_vals = vals.reshape(-1, wl.size)
    continua = np.empty_like(flat_vals)

    for idx, spec_vals in enumerate(flat_vals):
        hull_wl, hull_vals = _compute_convex_hull_continuum(wl, spec_vals)
        continua[idx] = _interpolate_continuum(wl, hull_wl, hull_vals)

    continuum_vals = continua.reshape(vals.shape)
    return Spectrum.from_reflectance(
        WavelengthGrid(wl),
        continuum_vals,
        units=spectrum.units,
        mask=spectrum.mask,
        meta=spectrum.meta.copy(),
    )


def _normalize_anchor_points(wl: np.ndarray, anchors: Iterable[float | int]) -> np.ndarray:
    anchor_points: list[float] = []
    for anchor in anchors:
        if isinstance(anchor, (list, tuple)) and len(anchor) == 2:
            anchor_points.extend(anchor)  # type: ignore[arg-type]
        else:
            anchor_points.append(anchor)  # type: ignore[arg-type]

    if not anchor_points:
        msg = "At least one anchor is required for anchor-based continua"
        raise ValueError(msg)

    normalized: list[float] = []
    for pt in anchor_points:
        if isinstance(pt, (int, np.integer)):
            idx = int(pt)
            if idx < 0 or idx >= wl.size:
                msg = "Anchor index is out of bounds for wavelength grid"
                raise IndexError(msg)
            normalized.append(float(wl[idx]))
        else:
            normalized.append(float(pt))

    normalized.extend([float(wl[0]), float(wl[-1])])
    return np.asarray(sorted(set(normalized)), dtype=np.float64)


def compute_anchor_continuum(spectrum: Spectrum, anchors: Iterable[float | int]) -> Spectrum:
    """Compute a piecewise-linear continuum using user-provided anchors.

    The anchors may be specified as wavelengths (nm) or integer band indices.
    Endpoints are implicitly included to ensure the continuum spans the entire
    spectrum. This mode is useful when shoulder locations are known from prior
    knowledge (e.g. laboratory measurements or configuration presets).
    """

    wl, vals = _validate_reflectance_spectrum(spectrum)
    anchor_points = _normalize_anchor_points(wl, anchors)

    flat_vals = vals.reshape(-1, wl.size)
    continua = np.empty_like(flat_vals)

    for idx, spec_vals in enumerate(flat_vals):
        anchor_vals = np.interp(anchor_points, wl, spec_vals)
        continua[idx] = _interpolate_continuum(wl, anchor_points, anchor_vals)

    continuum_vals = continua.reshape(vals.shape)
    return Spectrum.from_reflectance(
        WavelengthGrid(wl),
        continuum_vals,
        units=spectrum.units,
        mask=spectrum.mask,
        meta=spectrum.meta.copy(),
    )


def smooth_continuum(
    continuum: Spectrum, *, method: str = "poly", order: int = 2
) -> Spectrum:
    """Optionally smooth a continuum spectrum using polynomial or spline fits.

    Smoothing is off by default in :func:`build_continuum` but can be enabled to
    suppress high-frequency noise while retaining the broader continuum shape
    for both laboratory and overhead spectra.
    """

    wl, vals = _validate_reflectance_spectrum(continuum)
    flat_vals = vals.reshape(-1, wl.size)
    smoothed = np.empty_like(flat_vals)

    for idx, spec_vals in enumerate(flat_vals):
        if method == "poly":
            degree = min(order, max(1, wl.size - 1))
            coeffs = np.polyfit(wl, spec_vals, deg=degree)
            smoothed[idx] = np.polyval(coeffs, wl)
        elif method == "spline":
            if wl.size < 3:
                smoothed[idx] = spec_vals
            else:
                b, c, d = _natural_cubic_spline_coefficients(wl, spec_vals)
                smoothed[idx] = _evaluate_natural_cubic_spline(wl, spec_vals, b, c, d, wl)
        else:
            msg = f"Unknown smoothing option: {method}"
            raise ValueError(msg)

    smoothed_vals = smoothed.reshape(vals.shape)
    meta = continuum.meta.copy()
    meta.update({"continuum_smoothing_method": method, "continuum_smoothing_order": order})

    return Spectrum.from_reflectance(
        WavelengthGrid(wl),
        smoothed_vals,
        units=continuum.units,
        mask=continuum.mask,
        meta=meta,
    )


def build_continuum(
    spectrum: Spectrum,
    *,
    method: str = "convex_hull",
    anchors: Iterable[float | int] | None = None,
    smoothing: str | None = None,
    smoothing_order: int | None = None,
) -> np.ndarray:
    """Construct a continuum for a reflectance-like spectrum.

    Parameters
    ----------
    spectrum:
        Reflectance :class:`~alchemi.types.Spectrum` with strictly increasing
        wavelength grid (nm). Radiance or brightness temperature spectra are
        not supported.
    method:
        Continuum construction method. ``"convex_hull"`` builds the upper
        convex hull of (λ, R) and linearly interpolates between hull vertices.
        ``"anchors"`` uses user-provided anchor wavelengths or band indices to
        define piecewise linear segments.
    anchors:
        Anchors used by the ``"anchors"`` method. Values can be wavelengths or
        integer indices into the wavelength grid. Endpoints are implicitly
        included.
    smoothing:
        Optional smoothing applied to the continuum. ``"poly"`` fits a
        polynomial of degree ``smoothing_order`` (defaults to 2). ``"spline"``
        evaluates a natural cubic spline through the continuum points. Legacy
        ``"polyN"`` strings are still accepted.

    Notes
    -----
    The convex hull option is a safe default when absorption shoulders are not
    known in advance (e.g. arbitrary lab or airborne spectra). Anchor continua
    should be preferred when shoulder wavelengths or band indices are known a
    priori to prevent over-estimating shallow absorptions.
    """

    if method == "convex_hull":
        continuum_spec = compute_convex_hull_continuum(spectrum)
    elif method == "anchors":
        if anchors is None:
            msg = "anchors method requires anchors to be provided"
            raise ValueError(msg)
        continuum_spec = compute_anchor_continuum(spectrum, anchors)
    else:
        msg = f"Unknown continuum method: {method}"
        raise ValueError(msg)

    if smoothing:
        if smoothing.startswith("poly") and smoothing != "poly":
            match = re.match(r"poly(\d+)", smoothing)
            if not match:
                msg = f"Invalid polynomial smoothing specifier: {smoothing}"
                raise ValueError(msg)
            smoothing_order = int(match.group(1))
            smoothing_method = "poly"
        else:
            smoothing_method = smoothing
        order = 2 if smoothing_order is None else smoothing_order
        continuum_spec = smooth_continuum(continuum_spec, method=smoothing_method, order=order)

    return np.asarray(continuum_spec.values)


def continuum_remove(
    spectrum: Spectrum,
    *,
    method: str = "convex_hull",
    anchors: Iterable[float | int] | None = None,
    smoothing: str | None = None,
    continuum: Spectrum | None = None,
) -> Spectrum:
    """Apply continuum removal, returning a new reflectance spectrum."""

    if continuum is None:
        continuum_vals = build_continuum(
            spectrum, method=method, anchors=anchors, smoothing=smoothing
        )
    else:
        wl, cont_vals = _validate_reflectance_spectrum(continuum)
        if not np.allclose(wl, spectrum.wavelengths.nm):
            msg = "Continuum wavelength grid must match the spectrum"
            raise ValueError(msg)
        continuum_vals = cont_vals

    safe_continuum = np.clip(continuum_vals, _MIN_DENOM, None)
    removed_vals = spectrum.values / safe_continuum
    removed_vals = np.where(continuum_vals <= 0, np.nan, removed_vals)

    meta = spectrum.meta.copy()
    meta.update(
        {
            "continuum_removed": True,
            "continuum_method": method,
            "continuum_anchors": anchors,
            "continuum_smoothing": smoothing,
        }
    )

    return Spectrum.from_reflectance(
        spectrum.wavelengths,
        removed_vals,
        units=spectrum.units,
        mask=spectrum.mask,
        meta=meta,
    )


@dataclass(frozen=True)
class BandMetrics:
    depth: float
    area: float
    asymmetry: float
    lambda_center_nm: float
    lambda_left_nm: float
    lambda_right_nm: float


def _extract_value(wl: np.ndarray, vals: np.ndarray, target: float) -> float:
    return float(np.interp(target, wl, vals))


def compute_band_metrics(
    spectrum: Spectrum,
    *,
    lambda_left_nm: float,
    lambda_center_nm: float,
    lambda_right_nm: float,
    continuum: np.ndarray | Spectrum | None = None,
    method: str = "convex_hull",
    anchors: Iterable[float | int] | None = None,
    smoothing: str | None = None,
) -> BandMetrics:
    """Compute band depth, area, and asymmetry for a spectral feature."""

    if not (lambda_left_nm < lambda_center_nm < lambda_right_nm):
        msg = "Band bounds must satisfy left < center < right"
        raise ValueError(msg)

    wl = np.asarray(spectrum.wavelengths.nm, dtype=np.float64)
    vals = np.asarray(spectrum.values, dtype=np.float64)

    if vals.ndim != 1:
        msg = "Band metrics require a one-dimensional spectrum"
        raise ValueError(msg)

    if continuum is None:
        continuum_arr = build_continuum(
            spectrum, method=method, anchors=anchors, smoothing=smoothing
        )
    elif isinstance(continuum, Spectrum):
        wl_cont, continuum_arr = _validate_reflectance_spectrum(continuum)
        if not np.allclose(wl_cont, wl):
            msg = "Continuum wavelength grid must match the spectrum"
            raise ValueError(msg)
    else:
        continuum_arr = continuum
    safe_cont = np.clip(continuum_arr, _MIN_DENOM, None)
    removed = vals / safe_cont

    center_val = _extract_value(wl, vals, lambda_center_nm)
    cont_center = _extract_value(wl, continuum_arr, lambda_center_nm)
    depth = 0.0 if cont_center <= 0 else 1.0 - center_val / cont_center

    within = (wl >= lambda_left_nm) & (wl <= lambda_right_nm)
    wl_band = wl[within]
    removed_band = removed[within]
    if wl_band.size < 2:
        wl_band = np.asarray([lambda_left_nm, lambda_right_nm], dtype=np.float64)
        removed_band = np.asarray(
            [
                _extract_value(wl, removed, lambda_left_nm),
                _extract_value(wl, removed, lambda_right_nm),
            ]
        )

    band_area = np.trapezoid(1.0 - removed_band, wl_band)

    mid = lambda_center_nm
    left_mask = wl_band <= mid
    right_mask = wl_band >= mid
    left_area = (
        np.trapezoid(1.0 - removed_band[left_mask], wl_band[left_mask])
        if np.any(left_mask)
        else 0.0
    )
    right_area = (
        np.trapezoid(1.0 - removed_band[right_mask], wl_band[right_mask])
        if np.any(right_mask)
        else 0.0
    )
    asymmetry = left_area / right_area if right_area != 0 else np.nan

    return BandMetrics(
        depth=float(depth),
        area=float(band_area),
        asymmetry=float(asymmetry),
        lambda_center_nm=float(lambda_center_nm),
        lambda_left_nm=float(lambda_left_nm),
        lambda_right_nm=float(lambda_right_nm),
    )


def compute_band_depth(
    spectrum: Spectrum, lambda_center_nm: float, lambda_left_nm: float, lambda_right_nm: float
) -> float:
    """Backward-compatible single-point band depth using the shared utilities."""

    metrics = compute_band_metrics(
        spectrum,
        lambda_left_nm=lambda_left_nm,
        lambda_center_nm=lambda_center_nm,
        lambda_right_nm=lambda_right_nm,
    )
    return metrics.depth
