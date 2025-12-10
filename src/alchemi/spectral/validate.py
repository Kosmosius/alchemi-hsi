"""Runtime validators for spectral samples and spectra."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from alchemi.physics import rad_reflectance
from alchemi.registry.sensors import DEFAULT_SENSOR_REGISTRY, SensorSpec
from alchemi.types import (
    QuantityKind,
    RadianceUnits,
    ReflectanceUnits,
    Spectrum,
    TemperatureUnits,
)

DEFAULT_WAVELENGTH_PADDING_NM = 100.0
"""Slack applied when comparing spectrum wavelengths to a sensor range."""

DEFAULT_BAND_COUNT_TOLERANCE = 5
"""Allow a handful of bands to be trimmed relative to the nominal count."""

DEFAULT_BAND_CENTER_ATOL_NM = 25.0
"""Absolute tolerance for comparing band centre wavelengths to a sensor spec."""


def _require_strictly_increasing(arr: np.ndarray, *, name: str) -> None:
    if np.any(arr <= 0):
        msg = f"{name} must be positive"
        raise ValueError(msg)
    diffs = np.diff(arr)
    if np.any(diffs <= 0):
        msg = f"{name} must be strictly increasing"
        raise ValueError(msg)


def _maybe_get_sensor_spec(sample_sensor_id: str, sensor_spec: SensorSpec | None) -> SensorSpec | None:
    if sensor_spec is not None:
        return sensor_spec
    try:
        return DEFAULT_SENSOR_REGISTRY.get_sensor(sample_sensor_id)
    except Exception:
        return None


def validate_spectrum_physics(spectrum: Spectrum) -> None:
    """Validate basic physics invariants for a :class:`Spectrum` instance."""

    wavelengths = np.asarray(spectrum.wavelengths.nm, dtype=np.float64)
    _require_strictly_increasing(wavelengths, name="wavelength_nm")

    if spectrum.kind == QuantityKind.RADIANCE:
        if spectrum.units != RadianceUnits.W_M2_SR_NM:
            msg = "Radiance spectrum must use W·m⁻²·sr⁻¹·nm⁻¹ units"
            raise ValueError(msg)
    elif spectrum.kind == QuantityKind.REFLECTANCE:
        if spectrum.units != ReflectanceUnits.FRACTION:
            msg = "Reflectance spectrum must be a dimensionless fraction"
            raise ValueError(msg)
        min_val = float(np.nanmin(spectrum.values))
        max_val = float(np.nanmax(spectrum.values))
        if min_val < -0.1 or max_val > 5.0:
            msg = "Reflectance values fall outside plausible bounds"
            raise ValueError(msg)
    elif spectrum.kind == QuantityKind.BRIGHTNESS_T:
        if spectrum.units != TemperatureUnits.KELVIN:
            msg = "Brightness temperature must be expressed in Kelvin"
            raise ValueError(msg)
        if np.any(np.asarray(spectrum.values) <= 0):
            msg = "Brightness temperatures must be positive"
            raise ValueError(msg)


def _check_wavelength_range(
    *,
    spectrum: Spectrum,
    sensor_spec: SensorSpec,
    padding_nm: float,
) -> None:
    wl_min, wl_max = float(np.nanmin(spectrum.wavelengths.nm)), float(
        np.nanmax(spectrum.wavelengths.nm)
    )
    spec_min, spec_max = sensor_spec.wavelength_range_nm
    if wl_min < spec_min - padding_nm or wl_max > spec_max + padding_nm:
        msg = (
            "Spectrum wavelengths outside expected sensor range: "
            f"[{wl_min:.1f}, {wl_max:.1f}] vs [{spec_min:.1f}, {spec_max:.1f}]"
        )
        raise ValueError(msg)


def _check_band_count(*, spectrum: Spectrum, sensor_spec: SensorSpec) -> None:
    expected = int(sensor_spec.expected_band_count)
    observed = int(spectrum.band_count)
    slack = max(DEFAULT_BAND_COUNT_TOLERANCE, int(0.02 * expected))
    if abs(observed - expected) > slack:
        msg = (
            "Spectrum band count deviates from sensor expectation: "
            f"{observed} vs {expected} (tolerance ±{slack})"
        )
        raise ValueError(msg)


def _check_band_centers(
    *,
    band_centers: Iterable[float] | None,
    sensor_spec: SensorSpec,
    atol_nm: float,
) -> None:
    if band_centers is None:
        return
    centers = np.asarray(list(band_centers), dtype=np.float64)
    if centers.shape != sensor_spec.band_centers_nm.shape:
        msg = "Band centre shape mismatch with sensor specification"
        raise ValueError(msg)
    if not np.allclose(centers, sensor_spec.band_centers_nm, atol=atol_nm):
        msg = "Band centres differ significantly from sensor specification"
        raise ValueError(msg)


def _validate_radiance_geometry(sample: "Sample") -> None:
    # Avoid circular import at type-check time
    if sample.viewing_geometry is None:
        return
    solar_zenith = getattr(sample.viewing_geometry, "solar_zenith_deg", None)
    if solar_zenith is None:
        return
    dummy_esun = np.ones(sample.spectrum.band_count, dtype=np.float64)
    rad_reflectance._validate_shapes(sample.spectrum, dummy_esun, float(solar_zenith))


def validate_sample(sample: "Sample", sensor_spec: SensorSpec | None = None) -> None:
    """Validate a :class:`Sample` and its spectrum against physics invariants."""

    # Local import to sidestep circular dependency at module import time
    from alchemi.spectral.sample import Sample

    if not isinstance(sample, Sample):
        msg = "validate_sample expects a spectral Sample instance"
        raise TypeError(msg)

    validate_spectrum_physics(sample.spectrum)

    try:
        sample.validate()
    except Exception as exc:
        msg = "Sample structural validation failed"
        raise ValueError(msg) from exc

    spec = _maybe_get_sensor_spec(sample.sensor_id, sensor_spec)
    if spec is not None:
        _check_wavelength_range(
            spectrum=sample.spectrum, sensor_spec=spec, padding_nm=DEFAULT_WAVELENGTH_PADDING_NM
        )
        _check_band_count(spectrum=sample.spectrum, sensor_spec=spec)
        band_centers = getattr(sample.band_meta, "center_nm", None)
        _check_band_centers(
            band_centers=band_centers, sensor_spec=spec, atol_nm=DEFAULT_BAND_CENTER_ATOL_NM
        )

    if sample.spectrum.kind == QuantityKind.RADIANCE:
        _validate_radiance_geometry(sample)


__all__ = ["validate_spectrum_physics", "validate_sample"]
