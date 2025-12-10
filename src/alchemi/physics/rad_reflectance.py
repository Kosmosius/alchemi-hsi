"""Radiance and top-of-atmosphere reflectance conversions.

All helpers operate on canonical :class:`~alchemi.spectral.sample.Sample`
payloads: radiance in W·m⁻²·sr⁻¹·nm⁻¹ on a nanometre grid and reflectance as a
unitless fraction. The routines mirror Section-4 semantics and preserve masks
and metadata when moving between radiance and TOA reflectance.
"""

from __future__ import annotations

import logging

import numpy as np

from alchemi.physics import units
from alchemi.physics.solar import (
    earth_sun_distance_for_sample,
    esun_for_sample,
    get_reference_esun,
)
from alchemi.spectral.sample import Sample
from alchemi.types import QuantityKind, RadianceUnits, ReflectanceUnits, Spectrum, WavelengthGrid

__all__ = [
    "radiance_to_toa_reflectance",
    "toa_reflectance_to_radiance",
    "radiance_sample_to_toa_reflectance",
    "toa_reflectance_sample_to_radiance",
]

logger = logging.getLogger(__name__)

_REFLECTANCE_WARN_THRESHOLD = 1.5
_REFLECTANCE_FRACTION_WARN = 0.2
_REFLECTANCE_DIAGNOSTIC_THRESHOLD = 1.2


def _validate_shapes(
    spectrum: Spectrum, esun_band: np.ndarray, solar_zenith_deg: float
) -> tuple[np.ndarray, float]:
    wavelengths = np.asarray(spectrum.wavelengths.nm, dtype=np.float64)
    if esun_band.shape[0] != wavelengths.shape[0]:
        msg = "Solar irradiance array must match spectrum wavelengths"
        raise ValueError(msg)

    if solar_zenith_deg < 0 or solar_zenith_deg > 90:
        msg = "Solar zenith angle must be within [0, 90] degrees"
        raise ValueError(msg)

    cos_theta = float(np.cos(np.deg2rad(solar_zenith_deg)))
    if cos_theta <= 0:
        msg = "Cosine of solar zenith must be positive"
        raise ValueError(msg)

    return wavelengths, cos_theta


def _radiance_values_nm(spectrum: Spectrum) -> np.ndarray:
    if spectrum.kind != QuantityKind.RADIANCE:
        raise ValueError("Input spectrum must represent radiance")

    normalized_units = units.normalize_units(
        spectrum.units or RadianceUnits.W_M2_SR_NM, QuantityKind.RADIANCE
    )
    values, canonical_units = units.normalize_values_to_canonical(
        np.asarray(spectrum.values, dtype=np.float64),
        normalized_units,
        QuantityKind.RADIANCE,
    )
    if canonical_units != units.ValueUnits.RADIANCE_W_M2_SR_NM:
        msg = (
            "Radiance spectrum must be W·m⁻²·sr⁻¹·nm⁻¹; use alchemi.physics.units"
            " helpers to normalise."
        )
        raise ValueError(msg)
    return values


def _validate_reflectance_units(spectrum: Spectrum) -> None:
    if spectrum.kind != QuantityKind.TOA_REFLECTANCE:
        msg = "Input spectrum must represent TOA reflectance"
        raise ValueError(msg)
    normalized_units = units.normalize_units(
        spectrum.units or ReflectanceUnits.FRACTION, QuantityKind.TOA_REFLECTANCE
    )
    if normalized_units != units.ValueUnits.REFLECTANCE_FRACTION:
        msg = "Reflectance spectrum must be dimensionless fraction"
        raise ValueError(msg)


def _diagnose_reflectance(values: np.ndarray) -> None:
    diagnostic_fraction = float(np.mean(values > _REFLECTANCE_DIAGNOSTIC_THRESHOLD))
    warning_fraction = float(np.mean(values > _REFLECTANCE_WARN_THRESHOLD))
    max_r = float(np.nanmax(values)) if values.size else float("nan")
    if warning_fraction > _REFLECTANCE_FRACTION_WARN:
        logger.warning(
            "High TOA reflectance fraction: %.1f%% of bands exceed %.1f (max %.2f)",
            100.0 * warning_fraction,
            _REFLECTANCE_WARN_THRESHOLD,
            max_r,
        )
    else:
        logger.debug(
            "TOA reflectance diagnostics: %.1f%% above %.1f, max %.2f",
            100.0 * diagnostic_fraction,
            _REFLECTANCE_DIAGNOSTIC_THRESHOLD,
            max_r,
        )


def radiance_to_toa_reflectance(
    spectrum: Spectrum,
    *,
    esun_band: np.ndarray,
    d_au: float,
    solar_zenith_deg: float,
) -> Spectrum:
    """Convert radiance to TOA reflectance using Eq. (17) in Section 5.2."""

    radiance_values = _radiance_values_nm(spectrum)
    wavelengths, cos_theta = _validate_shapes(spectrum, esun_band, solar_zenith_deg)

    E_sun = np.asarray(esun_band, dtype=np.float64)
    distance2 = float(d_au) ** 2

    reflectance = (np.pi * radiance_values * distance2) / (E_sun * cos_theta)
    _diagnose_reflectance(reflectance)

    return Spectrum.from_toa_reflectance(
        WavelengthGrid(wavelengths),
        reflectance,
        units=ReflectanceUnits.FRACTION,
        mask=spectrum.mask,
        meta=dict(spectrum.meta),
    )


def toa_reflectance_to_radiance(
    spectrum: Spectrum,
    *,
    esun_band: np.ndarray,
    d_au: float,
    solar_zenith_deg: float,
) -> Spectrum:
    """Convert TOA reflectance to radiance using the inverse of Eq. (17)."""

    _validate_reflectance_units(spectrum)
    wavelengths, cos_theta = _validate_shapes(spectrum, esun_band, solar_zenith_deg)

    reflectance = np.asarray(spectrum.values, dtype=np.float64)
    E_sun = np.asarray(esun_band, dtype=np.float64)
    distance2 = float(d_au) ** 2

    radiance = (reflectance * E_sun * cos_theta) / (np.pi * distance2)

    return Spectrum.from_radiance(
        WavelengthGrid(wavelengths),
        radiance,
        units=RadianceUnits.W_M2_SR_NM,
        mask=spectrum.mask,
        meta=dict(spectrum.meta),
    )


def _resolve_solar_zenith(sample: Sample, override: float | None = None) -> float:
    if override is not None:
        return float(override)
    if getattr(sample.viewing_geometry, "solar_zenith_deg", None) is not None:
        return float(sample.viewing_geometry.solar_zenith_deg)
    if sample.ancillary and "solar_zenith_deg" in sample.ancillary:
        return float(sample.ancillary["solar_zenith_deg"])
    msg = "Solar zenith angle not available on sample; provide an override"
    raise ValueError(msg)


def _resolve_earth_sun_distance(sample: Sample, override: float | None = None) -> float:
    if override is not None:
        return float(override)
    if getattr(sample.viewing_geometry, "earth_sun_distance_au", None) is not None:
        return float(sample.viewing_geometry.earth_sun_distance_au)
    if sample.ancillary and "earth_sun_distance_au" in sample.ancillary:
        return float(sample.ancillary["earth_sun_distance_au"])
    return earth_sun_distance_for_sample(sample)


def _resolve_esun_band(sample: Sample, esun_ref: Spectrum | None = None) -> np.ndarray:
    if esun_ref is None:
        esun_ref = get_reference_esun()
    if esun_ref.band_count == sample.spectrum.band_count:
        return np.asarray(esun_ref.values, dtype=np.float64)
    return np.asarray(esun_for_sample(sample), dtype=np.float64)


def radiance_sample_to_toa_reflectance(
    sample: Sample,
    *,
    esun_ref: Spectrum | None = None,
    solar_zenith_deg: float | None = None,
    earth_sun_distance_au: float | None = None,
) -> Sample:
    """Return a new Sample with the spectrum converted to TOA reflectance."""

    esun_band = _resolve_esun_band(sample, esun_ref)
    solar_zenith = _resolve_solar_zenith(sample, solar_zenith_deg)
    d_au = _resolve_earth_sun_distance(sample, earth_sun_distance_au)

    reflectance = radiance_to_toa_reflectance(
        sample.spectrum, esun_band=esun_band, d_au=d_au, solar_zenith_deg=solar_zenith
    )
    return Sample(
        spectrum=reflectance,
        sensor_id=sample.sensor_id,
        acquisition_time=sample.acquisition_time,
        geo=sample.geo,
        viewing_geometry=sample.viewing_geometry,
        band_meta=sample.band_meta,
        srf_matrix=sample.srf_matrix,
        quality_masks=sample.quality_masks,
        ancillary=sample.ancillary,
    )


def toa_reflectance_sample_to_radiance(
    sample: Sample,
    *,
    esun_ref: Spectrum | None = None,
    solar_zenith_deg: float | None = None,
    earth_sun_distance_au: float | None = None,
) -> Sample:
    """Return a new Sample with TOA reflectance converted back to radiance."""

    esun_band = _resolve_esun_band(sample, esun_ref)
    solar_zenith = _resolve_solar_zenith(sample, solar_zenith_deg)
    d_au = _resolve_earth_sun_distance(sample, earth_sun_distance_au)

    radiance = toa_reflectance_to_radiance(
        sample.spectrum, esun_band=esun_band, d_au=d_au, solar_zenith_deg=solar_zenith
    )
    return Sample(
        spectrum=radiance,
        sensor_id=sample.sensor_id,
        acquisition_time=sample.acquisition_time,
        geo=sample.geo,
        viewing_geometry=sample.viewing_geometry,
        band_meta=sample.band_meta,
        srf_matrix=sample.srf_matrix,
        quality_masks=sample.quality_masks,
        ancillary=sample.ancillary,
    )
