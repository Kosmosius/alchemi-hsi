"""Radiance and top-of-atmosphere reflectance conversions.

This module contains the recommended entry points for moving between Level-1B
radiance and TOA reflectance as described in Section 5.2. Public functions:

* :func:`radiance_to_toa_reflectance` /
  :func:`radiance_sample_to_toa_reflectance` for forward conversion using Esun,
  Earth–Sun distance, and solar zenith.
* :func:`toa_reflectance_to_radiance` /
  :func:`toa_reflectance_sample_to_radiance` for the inverse.

Assumptions and units
---------------------
Radiance must be W·m⁻²·sr⁻¹·nm⁻¹ on a nanometre grid and reflectance is treated
as a dimensionless fraction. Callers are responsible for supplying band-matched
solar irradiance (E_sun) and solar geometry metadata; helpers will resample
Esun using the standard SRF machinery. Conversions assume the trusted SWIR
regime (``SWIRRegime.TRUSTED``) with moderate atmospheres—heavy haze, large PWV,
or high AOD can degrade accuracy.

The physics layer **does not perform atmospheric correction**—surface
reflectance retrieval must come from upstream L2A products produced by mission
pipelines or external radiative transfer tools. The helpers here are TOA-only.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from alchemi.physics import units
from alchemi.physics.rt_regime import SWIRRegime
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
_REFLECTANCE_STRICT_ENV = "ALCHEMI_REFLECTANCE_STRICT"
_HEAVY_ATMOS_WARNING = (
    "TOA reflectance approximation invoked outside trusted SWIR regime; heavy "
    "atmospheres can introduce domain shift and larger spectral distortions."
)


def _validate_shapes(
    spectrum: Spectrum, esun_band: np.ndarray, solar_zenith_deg: float
) -> tuple[np.ndarray, float]:
    wavelengths = np.asarray(spectrum.wavelengths.nm, dtype=np.float64)
    esun_array = np.asarray(esun_band, dtype=np.float64)
    if esun_array.ndim != 1 or esun_array.shape[0] != wavelengths.shape[0]:
        msg = "Solar irradiance array must match spectrum wavelengths and be 1-D"
        raise ValueError(msg)

    if solar_zenith_deg < 0 or solar_zenith_deg > 90:
        msg = "Solar zenith angle must be within [0, 90] degrees"
        raise ValueError(msg)

    cos_theta = float(np.cos(np.deg2rad(solar_zenith_deg)))
    if cos_theta <= 0 or np.isclose(cos_theta, 0.0, atol=1e-12):
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


def _validate_esun_and_distance(esun_band: np.ndarray, d_au: float) -> tuple[np.ndarray, float]:
    E_sun = np.asarray(esun_band, dtype=np.float64)
    if np.any(E_sun <= 0):
        msg = "Solar irradiance must be strictly positive"
        raise ValueError(msg)

    distance = float(d_au)
    if distance <= 0:
        msg = "Earth-Sun distance must be positive (AU)"
        raise ValueError(msg)
    return E_sun, distance ** 2


def _diagnose_reflectance(values: np.ndarray, *, strict: bool | None = None) -> None:
    diagnostic_fraction = float(np.mean(values > _REFLECTANCE_DIAGNOSTIC_THRESHOLD))
    warning_fraction = float(np.mean(values > _REFLECTANCE_WARN_THRESHOLD))
    min_r = float(np.nanmin(values)) if values.size else float("nan")
    max_r = float(np.nanmax(values)) if values.size else float("nan")
    strict_env = os.getenv(_REFLECTANCE_STRICT_ENV, "").strip().lower()
    effective_strict = bool(strict) or strict_env in {"1", "true", "yes", "on"}

    message = (
        "TOA reflectance diagnostics: %.1f%% above %.1f, %.1f%% above %.1f; "
        "range [%.3f, %.3f]"
    )
    if warning_fraction > _REFLECTANCE_FRACTION_WARN:
        if effective_strict:
            msg = message % (
                100.0 * diagnostic_fraction,
                _REFLECTANCE_DIAGNOSTIC_THRESHOLD,
                100.0 * warning_fraction,
                _REFLECTANCE_WARN_THRESHOLD,
                min_r,
                max_r,
            )
            raise ValueError(msg)

        logger.warning(
            message,
            100.0 * diagnostic_fraction,
            _REFLECTANCE_DIAGNOSTIC_THRESHOLD,
            100.0 * warning_fraction,
            _REFLECTANCE_WARN_THRESHOLD,
            min_r,
            max_r,
        )
    else:
        logger.debug(
            message,
            100.0 * diagnostic_fraction,
            _REFLECTANCE_DIAGNOSTIC_THRESHOLD,
            100.0 * warning_fraction,
            _REFLECTANCE_WARN_THRESHOLD,
            min_r,
            max_r,
        )


def _normalize_swir_regime(
    swir_regime: SWIRRegime | bool | str | None,
) -> SWIRRegime | None:
    if swir_regime is None:
        return None
    if isinstance(swir_regime, bool):
        return SWIRRegime.TRUSTED if swir_regime else SWIRRegime.HEAVY
    if isinstance(swir_regime, SWIRRegime):
        return swir_regime
    try:
        return SWIRRegime(str(swir_regime))
    except ValueError as exc:  # pragma: no cover - defensive for unexpected inputs
        msg = "swir_regime must be SWIRRegime, bool, or None"
        raise ValueError(msg) from exc


def _warn_if_heavy_atmosphere(swir_regime: SWIRRegime | bool | str | None) -> None:
    regime = _normalize_swir_regime(swir_regime)
    if regime is None:
        return
    if regime == SWIRRegime.HEAVY:
        logger.warning(_HEAVY_ATMOS_WARNING)


def radiance_to_toa_reflectance(
    spectrum: Spectrum,
    *,
    esun_band: np.ndarray,
    d_au: float,
    solar_zenith_deg: float,
    strict_diagnostics: bool | None = None,
    swir_regime: SWIRRegime | bool | str | None = None,
) -> Spectrum:
    """Convert radiance to **TOA** reflectance using Eq. (17) in Section 5.2.

    The approximation is physically justified for moderate atmospheres
    (``SWIRRegime.TRUSTED``) and intentionally stops short of estimating
    surface reflectance. Pass ``swir_regime`` when available to surface a
    warning under heavy conditions where external atmospheric correction is
    preferred.
    """

    radiance_values = _radiance_values_nm(spectrum)
    wavelengths, cos_theta = _validate_shapes(spectrum, esun_band, solar_zenith_deg)

    E_sun, distance2 = _validate_esun_and_distance(esun_band, d_au)

    _warn_if_heavy_atmosphere(swir_regime)

    reflectance = (np.pi * radiance_values * distance2) / (E_sun * cos_theta)
    _diagnose_reflectance(reflectance, strict=strict_diagnostics)

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
    swir_regime: SWIRRegime | bool | str | None = None,
) -> Spectrum:
    """Convert TOA reflectance to radiance using the inverse of Eq. (17).

    The inverse shares the same trusted SWIR regime assumptions as
    :func:`radiance_to_toa_reflectance` and will emit an optional warning when
    used under heavy atmospheric conditions. Surface reflectance should still be
    sourced from mission L2A products rather than inverting TOA in isolation.
    """

    _validate_reflectance_units(spectrum)
    wavelengths, cos_theta = _validate_shapes(spectrum, esun_band, solar_zenith_deg)

    reflectance = np.asarray(spectrum.values, dtype=np.float64)
    E_sun, distance2 = _validate_esun_and_distance(esun_band, d_au)

    _warn_if_heavy_atmosphere(swir_regime)

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


def _resolve_sample_swir_regime(
    sample: Sample, swir_regime: SWIRRegime | bool | str | None
) -> SWIRRegime | None:
    if swir_regime is not None:
        return _normalize_swir_regime(swir_regime)

    ancillary = getattr(sample, "ancillary", None) or {}
    stored_regime = ancillary.get("swir_regime")
    if stored_regime is None:
        return None

    try:
        return _normalize_swir_regime(stored_regime)
    except ValueError:
        return None


def radiance_sample_to_toa_reflectance(
    sample: Sample,
    *,
    esun_ref: Spectrum | None = None,
    solar_zenith_deg: float | None = None,
    earth_sun_distance_au: float | None = None,
    strict_diagnostics: bool | None = None,
    swir_regime: SWIRRegime | bool | str | None = None,
    warn_outside_trusted: bool = True,
) -> Sample:
    """Return a new Sample with the spectrum converted to TOA reflectance.

    When the sample (or override) carries a heavy SWIR regime tag, a warning is
    emitted by default to remind callers that the approximation is less
    reliable under those conditions. Set ``warn_outside_trusted=False`` to
    suppress the warning.
    """

    esun_band = _resolve_esun_band(sample, esun_ref)
    solar_zenith = _resolve_solar_zenith(sample, solar_zenith_deg)
    d_au = _resolve_earth_sun_distance(sample, earth_sun_distance_au)

    resolved_regime = _resolve_sample_swir_regime(sample, swir_regime)

    reflectance = radiance_to_toa_reflectance(
        sample.spectrum,
        esun_band=esun_band,
        d_au=d_au,
        solar_zenith_deg=solar_zenith,
        strict_diagnostics=strict_diagnostics,
        swir_regime=resolved_regime if warn_outside_trusted else None,
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
    swir_regime: SWIRRegime | bool | str | None = None,
    warn_outside_trusted: bool = True,
) -> Sample:
    """Return a new Sample with TOA reflectance converted back to radiance.

    Mirrors :func:`radiance_sample_to_toa_reflectance` and surfaces the same
    heavy-atmosphere warning unless ``warn_outside_trusted`` is disabled.
    """

    esun_band = _resolve_esun_band(sample, esun_ref)
    solar_zenith = _resolve_solar_zenith(sample, solar_zenith_deg)
    d_au = _resolve_earth_sun_distance(sample, earth_sun_distance_au)

    resolved_regime = _resolve_sample_swir_regime(sample, swir_regime)

    radiance = toa_reflectance_to_radiance(
        sample.spectrum,
        esun_band=esun_band,
        d_au=d_au,
        solar_zenith_deg=solar_zenith,
        swir_regime=resolved_regime if warn_outside_trusted else None,
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
