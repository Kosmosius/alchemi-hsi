"""Physics helper utilities for spectral conversions."""

from .planck import (
    C,
    H,
    K_B,
    bt_K_to_radiance,
    bt_to_radiance,
    radiance_to_bt,
    radiance_to_bt_K,
)
from .solar import get_E0_nm, sun_earth_factor
from .swir import band_depth, continuum_remove, radiance_to_reflectance, reflectance_to_radiance
from .swir_avirisng import (
    avirisng_bad_band_mask,
    radiance_to_reflectance_avirisng,
    reflectance_to_radiance_avirisng,
)

__all__ = [
    "band_depth",
    "avirisng_bad_band_mask",
    "bt_K_to_radiance",
    "bt_to_radiance",
    "C",
    "continuum_remove",
    "get_E0_nm",
    "H",
    "K_B",
    "radiance_to_bt",
    "radiance_to_bt_K",
    "radiance_to_reflectance",
    "radiance_to_reflectance_avirisng",
    "reflectance_to_radiance",
    "reflectance_to_radiance_avirisng",
    "sun_earth_factor",
]
