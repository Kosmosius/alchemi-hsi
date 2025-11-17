"""Physics helper utilities for spectral conversions."""

from .augment import augment_radiance, random_swirlike_atmosphere
from .planck import (
    K_B,
    C,
    H,
    bt_K_to_radiance,
    bt_to_radiance,
    radiance_to_bt,
    radiance_to_bt_K,
)
from .solar import get_E0_nm, sun_earth_factor
from .swir import (
    band_depth,
    continuum_remove,
    radiance_to_reflectance,
    reflectance_to_radiance,
)
from .swir_avirisng import (
    avirisng_bad_band_mask,
    radiance_to_reflectance_avirisng,
    reflectance_to_radiance_avirisng,
)

__all__ = [
    "K_B",
    "C",
    "H",
    "augment_radiance",
    "avirisng_bad_band_mask",
    "band_depth",
    "bt_K_to_radiance",
    "bt_to_radiance",
    "continuum_remove",
    "get_E0_nm",
    "radiance_to_bt",
    "radiance_to_bt_K",
    "radiance_to_reflectance",
    "radiance_to_reflectance_avirisng",
    "random_swirlike_atmosphere",
    "reflectance_to_radiance",
    "reflectance_to_radiance_avirisng",
    "sun_earth_factor",
]
