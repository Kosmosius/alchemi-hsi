"""Continuum removal and absorption feature metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from alchemi.types import QuantityKind, Spectrum, WavelengthGrid

__all__ = [
    "compute_convex_hull_continuum",
    "compute_band_depth",
]


def _upper_hull_indices(wavelengths: np.ndarray, values: np.ndarray) -> list[int]:
    """Compute indices forming the upper convex hull of (wavelength, value)."""

    points = list(zip(wavelengths, values))
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


def _interpolate_segments(anchors_wl: Iterable[float], anchors_val: Iterable[float], wl: np.ndarray) -> np.ndarray:
    anchors_wl_arr = np.asarray(list(anchors_wl), dtype=np.float64)
    anchors_val_arr = np.asarray(list(anchors_val), dtype=np.float64)
    return np.interp(wl, anchors_wl_arr, anchors_val_arr)


def compute_convex_hull_continuum(spectrum: Spectrum) -> Spectrum:
    """Estimate the continuum using the upper convex hull of a reflectance spectrum."""

    if spectrum.kind != QuantityKind.REFLECTANCE:
        raise ValueError("Continuum removal expects a reflectance spectrum")

    wl = np.asarray(spectrum.wavelengths.nm, dtype=np.float64)
    values = np.asarray(spectrum.values, dtype=np.float64)

    if wl.ndim != 1 or values.ndim != 1:
        raise ValueError("Spectrum must be one-dimensional")

    hull_idx = _upper_hull_indices(wl, values)
    hull_wl = wl[hull_idx]
    hull_vals = values[hull_idx]

    continuum = _interpolate_segments(hull_wl, hull_vals, wl)

    return Spectrum.from_reflectance(
        WavelengthGrid(wl),
        continuum,
        units=spectrum.units,
        mask=spectrum.mask,
        meta=spectrum.meta.copy(),
    )


def compute_band_depth(
    spectrum: Spectrum, lambda_center_nm: float, lambda_left_nm: float, lambda_right_nm: float
) -> float:
    """Compute band depth at ``lambda_center_nm`` using a straight-line continuum."""

    if spectrum.kind != QuantityKind.REFLECTANCE:
        raise ValueError("Band depth requires a reflectance spectrum")

    if not (lambda_left_nm < lambda_center_nm < lambda_right_nm):
        raise ValueError("Band bounds must satisfy left < center < right")

    wl = np.asarray(spectrum.wavelengths.nm, dtype=np.float64)
    vals = np.asarray(spectrum.values, dtype=np.float64)

    left_val = float(np.interp(lambda_left_nm, wl, vals))
    right_val = float(np.interp(lambda_right_nm, wl, vals))
    center_val = float(np.interp(lambda_center_nm, wl, vals))

    continuum_center = np.interp(
        lambda_center_nm,
        np.asarray([lambda_left_nm, lambda_right_nm], dtype=np.float64),
        np.asarray([left_val, right_val], dtype=np.float64),
    )

    if continuum_center <= 0:
        return 0.0

    depth = 1.0 - (center_val / continuum_center)
    return float(depth)
