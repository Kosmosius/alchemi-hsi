"""Radiance and top-of-atmosphere reflectance conversions."""

from __future__ import annotations

import numpy as np

from alchemi.types import (
    QuantityKind,
    RadianceUnits,
    ReflectanceUnits,
    Spectrum,
    WavelengthGrid,
)

__all__ = ["radiance_to_toa_reflectance", "toa_reflectance_to_radiance"]


def _validate_shapes(
    spectrum: Spectrum, solar_irradiance_nm: np.ndarray, solar_zenith_deg: float
) -> tuple[np.ndarray, float]:
    wavelengths = np.asarray(spectrum.wavelengths.nm, dtype=np.float64)
    if solar_irradiance_nm.shape[0] != wavelengths.shape[0]:
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


def radiance_to_toa_reflectance(
    spectrum: Spectrum,
    solar_zenith_deg: float,
    earth_sun_distance_au: float,
    solar_irradiance_nm: np.ndarray,
) -> Spectrum:
    """Convert radiance to TOA reflectance using the standard formula.

    R_TOA,λ = π L_TOA,λ d² / (E_sun,λ cos θ_s)
    """

    if spectrum.kind != QuantityKind.RADIANCE:
        raise ValueError("Input spectrum must represent radiance")

    wavelengths, cos_theta = _validate_shapes(
        spectrum, solar_irradiance_nm, solar_zenith_deg
    )

    L = np.asarray(spectrum.values, dtype=np.float64)
    E_sun = np.asarray(solar_irradiance_nm, dtype=np.float64)
    distance2 = float(earth_sun_distance_au) ** 2

    numerator = np.pi * L * distance2
    reflectance = numerator / (E_sun * cos_theta)

    return Spectrum.from_reflectance(
        WavelengthGrid(wavelengths),
        reflectance,
        units=ReflectanceUnits.FRACTION,
        mask=spectrum.mask,
        meta=spectrum.meta.copy(),
    )


def toa_reflectance_to_radiance(
    spectrum: Spectrum,
    solar_zenith_deg: float,
    earth_sun_distance_au: float,
    solar_irradiance_nm: np.ndarray,
) -> Spectrum:
    """Convert TOA reflectance to radiance."""

    if spectrum.kind != QuantityKind.REFLECTANCE:
        raise ValueError("Input spectrum must represent reflectance")

    wavelengths, cos_theta = _validate_shapes(
        spectrum, solar_irradiance_nm, solar_zenith_deg
    )

    reflectance = np.asarray(spectrum.values, dtype=np.float64)
    E_sun = np.asarray(solar_irradiance_nm, dtype=np.float64)
    distance2 = float(earth_sun_distance_au) ** 2

    radiance = (reflectance * E_sun * cos_theta) / (np.pi * distance2)

    return Spectrum.from_radiance(
        WavelengthGrid(wavelengths),
        radiance,
        units=RadianceUnits.W_M2_SR_NM,
        mask=spectrum.mask,
        meta=spectrum.meta.copy(),
    )
