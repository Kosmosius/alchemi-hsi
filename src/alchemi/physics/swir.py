"""SWIR single-layer helpers.

The utilities here implement the simple two-term radiance model consisting of a
bulk transmittance (``tau``) and an additive path radiance (``Lpath``). This is
**not** a full atmospheric correction: callers should obtain surface
reflectance from mission L2A products or external radiative transfer pipelines
and treat these helpers as TOA-facing approximations.
"""

from __future__ import annotations

import numpy as np

from alchemi.physics.continuum import (
    BandDefinition,
    build_continuum as _build_continuum,
    continuum_remove as _continuum_remove,
    compute_band_metrics as _compute_band_metrics,
)
from alchemi.types import Spectrum, WavelengthGrid


def reflectance_to_radiance(
    R: np.ndarray, E0: np.ndarray, cos_sun: float, tau: float, Lpath: float
) -> np.ndarray:
    """Approximate reflectance → TOA radiance using a 1-layer model.

    The model assumes a single effective transmittance term and additive path
    radiance; it does not recover surface reflectance and should be used for
    TOA-level simulations or quick-look conversions only.
    """

    return tau * (E0 * cos_sun / np.pi) * R + Lpath


def radiance_to_reflectance(
    L: np.ndarray, E0: np.ndarray, cos_sun: float, tau: float, Lpath: float
) -> np.ndarray:
    """Approximate TOA radiance → reflectance inversion.

    Uses the same single-layer assumptions as :func:`reflectance_to_radiance`
    and clips the result to [0, 1.5] for numerical stability. Surface
    reflectance retrieval should instead rely on L2A products.
    """

    denom = np.clip(tau * (E0 * cos_sun / np.pi), 1e-12, None)
    return np.clip((L - Lpath) / denom, 0.0, 1.5)


def continuum_remove(
    wavelength_nm: np.ndarray, reflectance: np.ndarray, left_nm: float, right_nm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Continuum removal convenience wrapper (no atmospheric modeling)."""

    wl = np.asarray(wavelength_nm, dtype=np.float64)
    refl = np.asarray(reflectance, dtype=np.float64)

    if wl.ndim != 1:
        raise ValueError("Wavelength grid must be one-dimensional")
    if refl.shape[-1] != wl.size:
        msg = "Last dimension of reflectance must match wavelength grid length"
        raise ValueError(msg)

    anchors = [(float(left_nm), float(right_nm))]
    spectrum = Spectrum.from_surface_reflectance(WavelengthGrid(wl), refl)
    continuum = _build_continuum(spectrum, method="anchors", anchors=anchors)
    removed = _continuum_remove(spectrum, method="anchors", anchors=anchors).values
    return continuum, removed


def band_depth(
    wavelength_nm: np.ndarray,
    reflectance: np.ndarray,
    center_nm: float,
    left_nm: float,
    right_nm: float,
) -> float:
    """Compute a continuum-removed band depth from surface reflectance."""

    wl = np.asarray(wavelength_nm, dtype=np.float64)
    refl = np.asarray(reflectance, dtype=np.float64)

    spectrum = Spectrum.from_surface_reflectance(WavelengthGrid(wl), refl)
    flat_vals = spectrum.values.reshape(-1, wl.size)
    depths = np.empty(flat_vals.shape[0], dtype=np.float64)

    for idx, spec_vals in enumerate(flat_vals):
        spec = Spectrum.from_reflectance(WavelengthGrid(wl), spec_vals)
        metrics = _compute_band_metrics(
            spec,
            band=BandDefinition(
                lambda_center_nm=float(center_nm),
                lambda_left_nm=float(left_nm),
                lambda_right_nm=float(right_nm),
            ),
            method="anchors",
            anchors=[(float(left_nm), float(right_nm))],
        )
        depths[idx] = metrics.depth

    return (
        float(depths.reshape(refl.shape[:-1]))
        if depths.size == 1
        else depths.reshape(refl.shape[:-1])
    )
