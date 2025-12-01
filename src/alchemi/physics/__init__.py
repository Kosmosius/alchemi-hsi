"""Physics helper utilities for spectral conversions."""

from .augment import augment_radiance, random_swirlike_atmosphere
from .planck import K_B, C, H, bt_K_to_radiance, bt_to_radiance, radiance_to_bt, radiance_to_bt_K
from .rad_reflectance import radiance_to_toa_reflectance, toa_reflectance_to_radiance
from .resampling import convolve_to_bands, generate_gaussian_srf, interpolate_to_centers, simulate_virtual_sensor
from .continuum import compute_band_depth, compute_convex_hull_continuum
from .rt_regime import classify_rt_regime
from .tes import tes_lwirt
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
    "classify_rt_regime",
    "compute_band_depth",
    "compute_convex_hull_continuum",
    "convolve_to_bands",
    "continuum_remove",
    "generate_gaussian_srf",
    "get_E0_nm",
    "interpolate_to_centers",
    "radiance_to_bt",
    "radiance_to_bt_K",
    "radiance_to_toa_reflectance",
    "radiance_to_reflectance",
    "radiance_to_reflectance_avirisng",
    "simulate_virtual_sensor",
    "random_swirlike_atmosphere",
    "reflectance_to_radiance",
    "reflectance_to_radiance_avirisng",
    "tes_lwirt",
    "toa_reflectance_to_radiance",
    "sun_earth_factor",
]
