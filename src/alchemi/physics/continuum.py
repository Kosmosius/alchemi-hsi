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
    "compute_band_depth",
    "compute_band_metrics",
    "compute_convex_hull_continuum",
    "continuum_remove",
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


def _interpolate_segments(
    anchors_wl: Iterable[float], anchors_val: Iterable[float], wl: np.ndarray
) -> np.ndarray:
    anchors_wl_arr = np.asarray(list(anchors_wl), dtype=np.float64)
    anchors_val_arr = np.asarray(list(anchors_val), dtype=np.float64)
    return np.interp(wl, anchors_wl_arr, anchors_val_arr)


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


def _anchor_continuum_points(
    wl: np.ndarray, vals: np.ndarray, anchors: list[tuple[float, float]] | None
) -> tuple[np.ndarray, np.ndarray]:
    anchor_points: list[float] = []
    if anchors:
        for left, right in anchors:
            anchor_points.extend([left, right])

    anchor_points.append(float(wl[0]))
    anchor_points.append(float(wl[-1]))
    anchor_points = sorted(set(anchor_points))

    anchor_vals = [float(np.interp(pt, wl, vals)) for pt in anchor_points]
    return np.asarray(anchor_points, dtype=np.float64), np.asarray(anchor_vals, dtype=np.float64)


def build_continuum(
    spectrum: Spectrum,
    *,
    method: str = "convex_hull",
    anchors: list[tuple[float, float]] | None = None,
    smoothing: str | None = None,
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
        ``"anchors"`` uses user-provided anchor pairs to define piecewise
        linear segments.
    anchors:
        Optional list of ``(λ_l, λ_r)`` anchor tuples in nanometres used by the
        ``"anchors"`` method.
    smoothing:
        Optional smoothing applied to the hull vertices. ``"polyN"`` fits a
        degree-N polynomial to hull points. ``"spline"`` evaluates a natural
        cubic spline through the hull vertices.
    """

    if spectrum.kind != QuantityKind.REFLECTANCE:
        raise ValueError("Continuum removal expects a reflectance-like spectrum")

    wl = np.asarray(spectrum.wavelengths.nm, dtype=np.float64)
    vals = np.asarray(spectrum.values, dtype=np.float64)

    if wl.ndim != 1:
        msg = "Spectrum must use a one-dimensional wavelength grid"
        raise ValueError(msg)
    if vals.shape[-1] != wl.size:
        msg = "Last dimension of values must match wavelength grid length"
        raise ValueError(msg)

    flat_vals = vals.reshape(-1, wl.size)
    continua = np.empty_like(flat_vals)

    for idx, spec_vals in enumerate(flat_vals):
        if method == "convex_hull":
            hull_wl, hull_vals = _compute_convex_hull_continuum(wl, spec_vals)
        elif method == "anchors":
            if anchors is None:
                msg = "anchors method requires anchors to be provided"
                raise ValueError(msg)
            hull_wl, hull_vals = _anchor_continuum_points(wl, spec_vals, anchors)
        else:
            msg = f"Unknown continuum method: {method}"
            raise ValueError(msg)

        if smoothing:
            if smoothing.startswith("poly"):
                match = re.match(r"poly(\d+)", smoothing)
                if not match:
                    msg = f"Invalid polynomial smoothing specifier: {smoothing}"
                    raise ValueError(msg)
                degree = int(match.group(1))
                degree = min(degree, max(1, hull_wl.size - 1))
                coeffs = np.polyfit(hull_wl, hull_vals, deg=degree)
                smoothed = np.polyval(coeffs, wl)
                continuum = smoothed
            elif smoothing == "spline":
                if hull_wl.size < 3:
                    continuum = _interpolate_segments(hull_wl, hull_vals, wl)
                else:
                    b, c, d = _natural_cubic_spline_coefficients(hull_wl, hull_vals)
                    continuum = _evaluate_natural_cubic_spline(hull_wl, hull_vals, b, c, d, wl)
            else:
                msg = f"Unknown smoothing option: {smoothing}"
                raise ValueError(msg)
        else:
            continuum = _interpolate_segments(hull_wl, hull_vals, wl)

        continua[idx] = continuum

    return continua.reshape(vals.shape)


def continuum_remove(
    spectrum: Spectrum,
    *,
    method: str = "convex_hull",
    anchors: list[tuple[float, float]] | None = None,
    smoothing: str | None = None,
) -> Spectrum:
    """Apply continuum removal, returning a new reflectance spectrum."""

    continuum = build_continuum(spectrum, method=method, anchors=anchors, smoothing=smoothing)
    safe_continuum = np.clip(continuum, _MIN_DENOM, None)
    removed_vals = spectrum.values / safe_continuum
    removed_vals = np.where(continuum <= 0, np.nan, removed_vals)

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
    continuum: np.ndarray | None = None,
    method: str = "convex_hull",
    anchors: list[tuple[float, float]] | None = None,
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

    continuum_arr = (
        build_continuum(spectrum, method=method, anchors=anchors, smoothing=smoothing)
        if continuum is None
        else continuum
    )
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


def compute_convex_hull_continuum(spectrum: Spectrum) -> Spectrum:
    """Backward-compatible wrapper returning a continuum spectrum."""

    continuum = build_continuum(spectrum, method="convex_hull")
    return Spectrum.from_reflectance(
        WavelengthGrid(np.asarray(spectrum.wavelengths.nm, dtype=np.float64)),
        continuum,
        units=spectrum.units,
        mask=spectrum.mask,
        meta=spectrum.meta.copy(),
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
