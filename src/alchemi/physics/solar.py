"""Solar irradiance and Earth-Sun distance utilities (ALCHEMI Section 5.2).

This module centralises solar-related helpers so the physics layer has a
single source of truth for:

* The reference extra-terrestrial solar irradiance spectrum (E₀ / Esun),
  stored on a fine wavelength grid and resampled via the same SRF machinery
  used for spectra.
* Projection of Esun onto sensor bandspaces using SRF convolutions or band
  centre interpolation when SRFs are unavailable.
* Earth-Sun distance computation for specific acquisition dates or samples.

The reference irradiance table is a lightweight, smooth approximation of the
ASTM G-173 AM0 spectrum sampled at 1 nm from 350-2500 nm, stored under
``resources/solar/esun_reference.csv``. Values are expressed as
W·m⁻²·nm⁻¹. Because the canonical :class:`~alchemi.types.Spectrum` only
recognises radiance/reflectance/BT quantity kinds, the Esun spectrum is tagged
with ``QuantityKind.RADIANCE`` while its metadata records that the values are
irradiance.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from alchemi.physics.resampling import convolve_to_bands, interpolate_to_centers
from alchemi.types import QuantityKind, RadianceUnits, Spectrum, SRFMatrix, WavelengthGrid
from alchemi.utils.integrate import np_integrate

try:  # Optional import for canonical spectral Sample/SRF types
    from alchemi.spectral.srf import SRFMatrix as SpectralSRFMatrix  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SpectralSRFMatrix = None  # type: ignore

__all__ = [
    "get_reference_esun",
    "project_esun_to_bands",
    "esun_for_sample",
    "earth_sun_distance_au",
    "earth_sun_distance_for_sample",
]


_ESUN_CACHE: Spectrum | None = None


def _resource_path() -> Path:
    return Path(__file__).resolve().parents[3] / "resources" / "solar" / "esun_reference.csv"


def get_reference_esun() -> Spectrum:
    """Return the high-resolution reference solar irradiance spectrum.

    The returned spectrum uses a strictly increasing nanometre grid with
    irradiance values in W·m⁻²·nm⁻¹. The quantity kind is stored as radiance for
    compatibility with :class:`~alchemi.types.Spectrum`; ``meta['quantity']`` is
    set to ``"irradiance"`` to prevent confusion with true sensor radiance.
    """

    global _ESUN_CACHE
    if _ESUN_CACHE is None:
        path = _resource_path()
        try:
            wavelengths, irradiance = np.loadtxt(
                path, delimiter=",", skiprows=1, dtype=np.float64, unpack=True
            )
        except OSError as exc:  # pragma: no cover - runtime error path
            msg = f"Failed to load reference Esun table from {path}"
            raise FileNotFoundError(msg) from exc
        if wavelengths.ndim != 1:
            msg = "Reference Esun wavelengths must be 1-D"
            raise ValueError(msg)
        if not np.all(np.diff(wavelengths) > 0):
            msg = "Reference Esun wavelengths must be strictly increasing"
            raise ValueError(msg)
        if np.any(irradiance <= 0):
            msg = "Reference Esun irradiance must be positive"
            raise ValueError(msg)

        _ESUN_CACHE = Spectrum(
            wavelengths=WavelengthGrid(wavelengths),
            values=irradiance,
            kind=QuantityKind.RADIANCE,
            units=RadianceUnits.W_M2_SR_NM,
            meta={"quantity": "irradiance", "source": "ASTM G-173 inspired"},
        )

    return _ESUN_CACHE


def _convert_spectral_srf(
    srf_matrix: Any, *, centers_hint: Iterable[float] | None = None
) -> SRFMatrix:
    if SpectralSRFMatrix is None or not isinstance(srf_matrix, SpectralSRFMatrix):
        raise TypeError("Expected a spectral.SRFMatrix for conversion")

    nm = np.asarray(srf_matrix.wavelength_nm, dtype=np.float64)
    resp = np.asarray(srf_matrix.matrix, dtype=np.float64)
    if resp.ndim != 2:
        msg = "SRF matrix must be 2-D (bands x wavelengths)"
        raise ValueError(msg)

    band_count = resp.shape[0]
    centers: list[float] = []
    if centers_hint is not None:
        centers_array = np.asarray(list(centers_hint), dtype=np.float64)
        if centers_array.shape[0] == band_count:
            centers = centers_array.tolist()
    if not centers:
        for row in resp:
            area = float(np_integrate(row, nm))
            if area <= 0:
                raise ValueError("SRF rows must integrate to a positive area")
            centers.append(float(np_integrate(row * nm, nm) / area))

    bands_nm = [nm for _ in range(band_count)]
    bands_resp = [resp[idx, :] for idx in range(band_count)]
    srf = SRFMatrix(
        sensor="unknown", centers_nm=np.asarray(centers), bands_nm=bands_nm, bands_resp=bands_resp
    )
    return srf.normalize_rows_trapz()


def project_esun_to_bands(
    esun: Spectrum, srf_matrix: SRFMatrix | Any, *, centers_hint: Iterable[float] | None = None
) -> np.ndarray:
    """Project high-resolution Esun to sensor bands via SRF convolution."""

    if isinstance(srf_matrix, SRFMatrix):
        srf = srf_matrix.normalize_rows_trapz()
    elif SpectralSRFMatrix is not None and isinstance(srf_matrix, SpectralSRFMatrix):
        srf = _convert_spectral_srf(srf_matrix, centers_hint=centers_hint)
    else:
        raise TypeError("Unsupported SRF matrix type for Esun projection")

    convolved = convolve_to_bands(esun, srf)
    return np.asarray(convolved.values, dtype=np.float64)


def esun_for_sample(sample: Any, *, mode: str = "srf") -> np.ndarray:
    """Return Esun projected to a sample's bandspace.

    Parameters
    ----------
    sample:
        Sample carrying SRF information or band centres. Both the canonical
        :class:`alchemi.spectral.sample.Sample` and legacy objects exposing
        ``srf_matrix``/``band_meta`` attributes are supported.
    mode:
        ``"srf"`` (default) uses SRF convolution when available; ``"interp"``
        forces interpolation at band centres.
    """

    esun = get_reference_esun()
    centers = None
    if hasattr(sample, "band_meta") and sample.band_meta is not None:
        centers = getattr(sample.band_meta, "center_nm", None)

    if mode == "srf" and hasattr(sample, "srf_matrix") and sample.srf_matrix is not None:
        return project_esun_to_bands(esun, sample.srf_matrix, centers_hint=centers)

    if centers is None and hasattr(sample, "spectrum"):
        centers = getattr(sample.spectrum, "wavelength_nm", None)
    if centers is None:
        msg = "Cannot determine band centres for Esun projection"
        raise ValueError(msg)

    interpolated = interpolate_to_centers(esun, np.asarray(centers, dtype=np.float64))
    return np.asarray(interpolated.values, dtype=np.float64)


def earth_sun_distance_au(
    date: datetime | np.datetime64 | None = None, *, doy: int | None = None
) -> float:
    """Return Earth-Sun distance in astronomical units for a date or day-of-year."""

    if doy is None:
        if date is None:
            date = datetime.utcnow()
        if isinstance(date, np.datetime64):
            date = datetime.utcfromtimestamp(date.astype("datetime64[s]").astype(int))
        doy = int(date.timetuple().tm_yday)

    doy = int(doy)
    if doy < 1 or doy > 366:
        msg = "Day of year must be within [1, 366]"
        raise ValueError(msg)

    g = 2.0 * np.pi * (doy - 1) / 365.0
    distance = (
        1.00014
        - 0.01671 * np.cos(g)
        - 0.00014 * np.cos(2 * g)
        + 0.000283 * np.sin(g)
        + 0.000033 * np.sin(2 * g)
    )
    return float(distance)


def earth_sun_distance_for_sample(sample: Any) -> float:
    """Extract Earth-Sun distance from sample metadata or observation time."""

    if (
        hasattr(sample, "viewing_geometry")
        and getattr(sample.viewing_geometry, "earth_sun_distance_au", None) is not None
    ):
        return float(sample.viewing_geometry.earth_sun_distance_au)  # type: ignore[return-value]

    if hasattr(sample, "ancillary"):
        ancillary = getattr(sample, "ancillary") or {}
        if isinstance(ancillary, dict) and "earth_sun_distance_au" in ancillary:
            return float(ancillary["earth_sun_distance_au"])

    acquisition_time = None
    if hasattr(sample, "acquisition_time"):
        acquisition_time = getattr(sample, "acquisition_time")
    elif hasattr(sample, "meta"):
        acquisition_time = getattr(sample, "meta", {}).get("acquisition_time")

    if acquisition_time is not None:
        return earth_sun_distance_au(acquisition_time)

    msg = "Earth-Sun distance unavailable: provide viewing_geometry, ancillary, or acquisition_time"
    raise ValueError(msg)
