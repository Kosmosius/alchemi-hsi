"""Continuum removal and absorption feature metrics.

This module implements the continuum-removal pipeline described in Section 5.4
of the ALCHEMI spec. It provides helpers to derive convex-hull or anchor-based
continua, perform continuum normalisation, and quantify absorption features via
band depth, area, asymmetry, and composite ratios. All functions expect
reflectance-like spectra (surface, TOA, or laboratory) on a monotonically
increasing nanometre grid and treat reflectance values as dimensionless
fractions. Public entry points:

* :func:`continuum_remove` for dividing a spectrum by its continuum.
* :func:`compute_band_depth`, :func:`compute_band_area`,
  :func:`compute_band_asymmetry`, and :func:`compute_band_metrics` for band
  analytics on continuum-removed spectra.
* :func:`build_continuum` / :func:`compute_anchor_continuum` for explicitly
  constructing continua prior to removal.

Continuum estimation assumes broad, concave absorption features; convex hulls
are used by default with optional spline smoothing. Values are not clipped, so
input validation (e.g., non-negative reflectance) should be handled upstream.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from alchemi.types import BandDefinition, QuantityKind, Spectrum, WavelengthGrid

__all__ = [
    "BandMetrics",
    "BandDefinition",
    "build_continuum",
    "compute_anchor_continuum",
    "compute_band_depth",
    "compute_band_area",
    "compute_band_asymmetry",
    "compute_composite_depth_ratio",
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
    continuum: Spectrum | np.ndarray | None = None,
) -> Spectrum:
    """Apply continuum removal, returning a new reflectance spectrum."""

    if continuum is None:
        continuum_vals = build_continuum(
            spectrum, method=method, anchors=anchors, smoothing=smoothing
        )
    elif isinstance(continuum, Spectrum):
        wl, cont_vals = _validate_reflectance_spectrum(continuum)
        if not np.allclose(wl, spectrum.wavelengths.nm):
            msg = "Continuum wavelength grid must match the spectrum"
            raise ValueError(msg)
        continuum_vals = cont_vals
    else:
        continuum_vals = np.asarray(continuum, dtype=np.float64)
        if continuum_vals.shape != spectrum.values.shape:
            msg = "Continuum array shape must match the spectrum values"
            raise ValueError(msg)

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
    band: BandDefinition

    @property
    def lambda_center_nm(self) -> float:  # pragma: no cover - trivial
        return self.band.lambda_center_nm

    @property
    def lambda_left_nm(self) -> float:  # pragma: no cover - trivial
        return self.band.lambda_left_nm

    @property
    def lambda_right_nm(self) -> float:  # pragma: no cover - trivial
        return self.band.lambda_right_nm


def _extract_value(wl: np.ndarray, vals: np.ndarray, target: float) -> float:
    valid = np.isfinite(vals)
    if not np.any(valid):
        return float("nan")
    return float(np.interp(target, wl[valid], vals[valid]))


def _interp_scalar_all(wl: np.ndarray, vals: np.ndarray, target: float) -> np.ndarray:
    """Vectorised scalar interpolation along the spectral axis.

    ``vals`` may contain arbitrary leading spatial dimensions with wavelengths on
    the last axis. A single target wavelength is interpolated for all spectra
    in one pass, avoiding Python loops over pixels.
    """

    if vals.shape[-1] != wl.size:
        msg = "values last dimension must match wavelength grid length"
        raise ValueError(msg)

    idx = np.searchsorted(wl, target, side="left")
    idx = np.clip(idx, 1, wl.size - 1)

    x0 = wl[idx - 1]
    x1 = wl[idx]
    y0 = vals[..., idx - 1]
    y1 = vals[..., idx]

    weight = (target - x0) / (x1 - x0)
    return y0 + weight * (y1 - y0)


def _mask_values(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    return np.where(mask, np.nan, values) if mask is not None else values


def _reshape_output(values: list[float], shape: tuple[int, ...]) -> float | np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(shape)
    return float(arr) if arr.shape == () else arr


def _reshape_metrics(values: list[BandMetrics], shape: tuple[int, ...]) -> BandMetrics | np.ndarray:
    arr = np.asarray(values, dtype=object).reshape(shape)
    return arr.item() if arr.shape == () else arr


def _ensure_continuum_removed(
    spectrum: Spectrum,
    *,
    continuum: np.ndarray | Spectrum | None,
    method: str,
    anchors: Iterable[float | int] | None,
    smoothing: str | None,
) -> Spectrum:
    if spectrum.meta.get("continuum_removed"):
        return spectrum
    return continuum_remove(
        spectrum,
        method=method,
        anchors=anchors,
        smoothing=smoothing,
        continuum=continuum if isinstance(continuum, (Spectrum, np.ndarray)) else None,
    )


def _band_window(wl: np.ndarray, values: np.ndarray, band: BandDefinition) -> tuple[np.ndarray, np.ndarray]:
    within = (wl >= band.lambda_left_nm) & (wl <= band.lambda_right_nm)
    wl_band = wl[within]
    vals_band = values[..., within]
    if wl_band.size < 2:
        wl_band = np.asarray([band.lambda_left_nm, band.lambda_right_nm], dtype=np.float64)
        left = _interp_scalar_all(wl, values, band.lambda_left_nm)
        right = _interp_scalar_all(wl, values, band.lambda_right_nm)
        vals_band = np.stack([left, right], axis=-1)
    return wl_band, np.asarray(vals_band)


def _band_depth_from_removed(continuum_removed: Spectrum, band: BandDefinition) -> float | np.ndarray:
    wl, vals = _validate_reflectance_spectrum(continuum_removed)
    vals = _mask_values(vals, continuum_removed.mask)
    center_vals = _interp_scalar_all(wl, vals, band.lambda_center_nm)
    return 1.0 - center_vals


def _compute_band_depth_removed(continuum_removed: Spectrum, band: BandDefinition) -> float | np.ndarray:
    return _band_depth_from_removed(continuum_removed, band)


def compute_band_area(continuum_removed: Spectrum, band: BandDefinition) -> float | np.ndarray:
    """Integrate band area over the wavelength interval."""

    wl, vals = _validate_reflectance_spectrum(continuum_removed)
    vals = _mask_values(vals, continuum_removed.mask)
    wl_band, removed_band = _band_window(wl, vals, band)
    valid = np.asarray(np.count_nonzero(np.isfinite(removed_band), axis=-1))
    if wl_band.size < 2 or np.min(valid, initial=2) < 2:
        return np.full(vals.shape[:-1], float("nan"))

    area = np.trapezoid(1.0 - removed_band, wl_band, axis=-1)
    return area


def compute_band_asymmetry(continuum_removed: Spectrum, band: BandDefinition) -> float | np.ndarray:
    """Compute left/right area ratio for a band."""

    wl, vals = _validate_reflectance_spectrum(continuum_removed)
    vals = _mask_values(vals, continuum_removed.mask)
    wl_band, removed_band = _band_window(wl, vals, band)
    valid = np.asarray(np.count_nonzero(np.isfinite(removed_band), axis=-1))
    if wl_band.size < 2 or np.min(valid, initial=2) < 2:
        return np.full(vals.shape[:-1], float("nan"))

    left_mask = wl_band <= band.lambda_center_nm
    right_mask = wl_band >= band.lambda_center_nm

    left_area = np.trapezoid(1.0 - removed_band[..., left_mask], wl_band[left_mask], axis=-1)
    right_area = np.trapezoid(1.0 - removed_band[..., right_mask], wl_band[right_mask], axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        asymmetry = left_area / right_area

    asymmetry = np.where(right_area == 0, np.nan, asymmetry)
    return asymmetry


def compute_band_metrics(
    spectrum: Spectrum,
    *,
    band: BandDefinition,
    continuum: np.ndarray | Spectrum | None = None,
    method: str = "convex_hull",
    anchors: Iterable[float | int] | None = None,
    smoothing: str | None = None,
) -> BandMetrics | np.ndarray:
    """Compute depth, area, and asymmetry for a band.

    If ``spectrum`` is already continuum-removed (``meta['continuum_removed']``),
    metrics are computed directly. Otherwise a continuum is built using the
    provided options before applying the metrics to the continuum-removed
    spectrum. Results are shaped to match any leading spatial dimensions.
    """

    removed = _ensure_continuum_removed(
        spectrum,
        continuum=continuum,
        method=method,
        anchors=anchors,
        smoothing=smoothing,
    )

    depth = _compute_band_depth_removed(removed, band)
    area = compute_band_area(removed, band)
    asymmetry = compute_band_asymmetry(removed, band)

    if isinstance(depth, float):
        return BandMetrics(depth=float(depth), area=float(area), asymmetry=float(asymmetry), band=band)

    metrics: list[BandMetrics] = []
    for d, a, asym in zip(depth.reshape(-1), np.asarray(area).reshape(-1), np.asarray(asymmetry).reshape(-1), strict=False):
        metrics.append(BandMetrics(depth=float(d), area=float(a), asymmetry=float(asym), band=band))

    return _reshape_metrics(metrics, np.shape(depth))


def compute_band_depth(
    spectrum: Spectrum,
    lambda_center_nm: float | None = None,
    lambda_left_nm: float | None = None,
    lambda_right_nm: float | None = None,
    band: BandDefinition | None = None,
    continuum: np.ndarray | Spectrum | None = None,
    method: str = "convex_hull",
    anchors: Iterable[float | int] | None = None,
    smoothing: str | None = None,
) -> float | np.ndarray:
    """Backward-compatible single-point band depth.

    The preferred usage is ``compute_band_depth(continuum_removed, band=...)``
    where ``continuum_removed`` is a continuum-normalised reflectance spectrum.
    Legacy positional bounds are still accepted for compatibility with existing
    callers.
    """

    band_def = band
    if band_def is None:
        if None in {lambda_center_nm, lambda_left_nm, lambda_right_nm}:
            msg = "Either a BandDefinition or explicit wavelengths must be provided"
            raise ValueError(msg)
        band_def = BandDefinition(
            lambda_center_nm=float(lambda_center_nm),
            lambda_left_nm=float(lambda_left_nm),
            lambda_right_nm=float(lambda_right_nm),
        )

    removed = _ensure_continuum_removed(
        spectrum,
        continuum=continuum,
        method=method,
        anchors=anchors,
        smoothing=smoothing,
    )
    return _band_depth_from_removed(removed, band_def)


def compute_composite_depth_ratio(
    band_metrics: Mapping[str, BandMetrics],
    numerator_weights: Mapping[str, float],
    denominator_weights: Mapping[str, float] | None = None,
) -> float:
    """Compute a weighted depth ratio used by multi-band diagnostics."""

    def _weighted_sum(weights: Mapping[str, float]) -> float:
        total = 0.0
        for key, weight in weights.items():
            if key not in band_metrics:
                msg = f"Band metric {key!r} not provided"
                raise KeyError(msg)
            total += weight * float(band_metrics[key].depth)
        return total

    numerator = _weighted_sum(numerator_weights)
    if denominator_weights is None:
        return numerator

    denominator = _weighted_sum(denominator_weights)
    return float(np.nan) if denominator == 0 else float(numerator / denominator)
