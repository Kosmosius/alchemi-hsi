"""HyTES longwave spectral response matrix generation.

The original HyTES instrument spectral response functions (SRFs) are not bundled
with the open-source portion of ALCHEMI for licensing reasons.  To keep the
processing pipeline working end-to-end we provide a lightweight approximation
that mimics the instrument behaviour: each band is modelled as a Gaussian
passband centred on the published HyTES wavelength grid.  The Gaussians are
constructed using the band spacing as the full-width at half-maximum (FWHM).
This approximation preserves energy under a flat spectrum after normalisation
but slightly extends beyond the nominal 7.5–12 μm range; that overshoot is
intentional so that the trapezoidal normalisation retains unit area.

The helper normalises each SRF with :func:`numpy.trapezoid` and stamps a stable
cache key so downstream caches can detect updates.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Tuple

import numpy as np

from alchemi.types import SRFMatrix
from alchemi_hsi.io.hytes import HYTES_WAVELENGTHS_NM

_SENSOR_NAME = "HyTES"
_DEFAULT_VERSION = "v1"
_CACHE: Dict[Tuple[str, str], SRFMatrix] = {}


def hytes_srf_matrix(*, version: str = _DEFAULT_VERSION) -> SRFMatrix:
    """Return a cached, trapz-normalised SRF matrix for the HyTES sensor.

    Parameters
    ----------
    version:
        Version tag describing the underlying SRF approximation.  Only ``"v1"``
        is recognised at the moment.

    Returns
    -------
    SRFMatrix
        Spectral response functions expressed on a nanometre grid where each
        row integrates to unity.
    """

    key = (_SENSOR_NAME, version)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    if version != _DEFAULT_VERSION:
        raise ValueError(f"Unsupported HyTES SRF version: {version!r}")

    centers_nm = HYTES_WAVELENGTHS_NM.astype(np.float64, copy=True)
    bands_nm, bands_resp = _approximate_gaussian_bands(centers_nm)

    srf = SRFMatrix(_SENSOR_NAME, centers_nm, bands_nm, bands_resp, version=version)
    srf = srf.normalize_trapz()
    srf.cache_key = _make_cache_key(srf)

    _CACHE[key] = srf
    return srf


def _approximate_gaussian_bands(centers_nm: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Construct Gaussian-like SRFs centred on ``centers_nm``.

    The Gaussian width is derived from the median band spacing which provides a
    smooth approximation that closely mimics the instrument's resolving power.
    """

    centers_nm = np.asarray(centers_nm, dtype=np.float64)
    if centers_nm.ndim != 1:
        raise ValueError("centers_nm must be 1-D")

    diffs = np.diff(centers_nm)
    if diffs.size == 0:
        raise ValueError("centers_nm must contain at least two elements")
    spacing = float(np.median(diffs))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("HyTES centres must be strictly increasing")

    # Convert FWHM to Gaussian sigma: FWHM = 2 * sqrt(2 ln 2) * sigma
    sigma = spacing / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    offsets = np.linspace(-3.0 * sigma, 3.0 * sigma, 121, dtype=np.float64)

    bands_nm: list[np.ndarray] = []
    bands_resp: list[np.ndarray] = []
    base_profile = np.exp(-0.5 * (offsets / sigma) ** 2).astype(np.float64, copy=False)

    for center in centers_nm:
        nm = (center + offsets).astype(np.float64, copy=True)
        response = base_profile.astype(np.float64, copy=True)
        bands_nm.append(nm)
        bands_resp.append(response)
    return bands_nm, bands_resp


def _make_cache_key(srf: SRFMatrix) -> str:
    hasher = hashlib.sha1()
    hasher.update(srf.sensor.encode("utf-8"))
    hasher.update(srf.version.encode("utf-8"))
    hasher.update(np.asarray(srf.centers_nm, dtype=np.float64).tobytes())
    for nm, resp in zip(srf.bands_nm, srf.bands_resp):
        hasher.update(np.asarray(nm, dtype=np.float64).tobytes())
        hasher.update(np.asarray(resp, dtype=np.float64).tobytes())
    digest = hasher.hexdigest()[:12]
    return f"{srf.sensor}:{srf.version}:{digest}"


__all__ = ["hytes_srf_matrix"]
