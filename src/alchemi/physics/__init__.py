"""User-facing entry points for the physics layer.

The physics package centralises the spectral conversions described in Section 5
of the ALCHEMI design doc. Importing from ``alchemi.physics`` surfaces the
recommended APIs without requiring callers to know the underlying modules:

* Planck and brightness-temperature conversions (:mod:`.planck`).
* Radiance ↔ TOA reflectance utilities (:mod:`.rad_reflectance`).
* Solar irradiance tables and Earth–Sun distance helpers (:mod:`.solar`).
* SWIR radiative-transfer regime tagging (:mod:`.rt_regime`).
* Continuum removal and band metrics (:mod:`.continuum`).
* Spectral resampling and virtual sensor generation (:mod:`.resampling`).
* LWIR TES scaffolding and emissivity proxies (:mod:`.tes`).

Most users should start with :func:`radiance_to_bt`,
:func:`radiance_to_toa_reflectance`, :func:`convolve_to_bands`, and
:func:`classify_swir_regime`. Lower-level helpers remain available for advanced
pipelines.
"""

from .augment import augment_radiance, random_swirlike_atmosphere
from .continuum import (
    BandDefinition,
    BandMetrics,
    build_continuum,
    compute_band_area,
    compute_band_asymmetry,
    compute_band_depth,
    compute_band_metrics,
    compute_composite_depth_ratio,
    compute_convex_hull_continuum,
    continuum_remove,
)
from .planck import (
    K_B,
    C,
    H,
    band_averaged_radiance,
    bt_K_to_radiance,
    bt_sample_to_radiance_sample,
    bt_spectrum_to_radiance,
    bt_to_radiance,
    inverse_planck_central_lambda,
    planck_radiance_wavelength,
    radiance_sample_to_bt_sample,
    radiance_spectrum_to_bt,
    radiance_to_bt,
    radiance_to_bt_K,
)
from .rad_reflectance import (
    radiance_sample_to_toa_reflectance,
    radiance_to_toa_reflectance,
    toa_reflectance_sample_to_radiance,
    toa_reflectance_to_radiance,
)
from .resampling import (
    convolve_to_bands,
    generate_gaussian_srf,
    interpolate_to_centers,
    SyntheticSensorConfig,
    simulate_virtual_sensor,
)
from .rt_regime import (
    SWIRRegime,
    attach_swir_regime,
    classify_rt_regime,
    classify_swir_regime,
    swir_regime_for_sample,
    swir_regime_for_scene,
    trusted_swir,
)
from .solar import (
    earth_sun_distance_au,
    earth_sun_distance_for_sample,
    esun_for_sample,
    get_reference_esun,
    project_esun_to_bands,
)
from .swir import band_depth, radiance_to_reflectance, reflectance_to_radiance
from .swir_avirisng import (
    avirisng_bad_band_mask,
    radiance_to_reflectance_avirisng,
    reflectance_to_radiance_avirisng,
)
from .tes import (
    bt_spectrum_to_radiance_spectrum,
    compute_lwir_emissivity_proxy,
    lwir_pipeline_for_sample,
    radiance_spectrum_to_bt_spectrum,
    tes_lwirt,
)

__all__ = [
    # Core constants
    "K_B",
    "C",
    "H",
    # Planck / BT conversions
    "band_averaged_radiance",
    "bt_K_to_radiance",
    "bt_sample_to_radiance_sample",
    "bt_spectrum_to_radiance",
    "bt_to_radiance",
    "inverse_planck_central_lambda",
    "planck_radiance_wavelength",
    "radiance_sample_to_bt_sample",
    "radiance_spectrum_to_bt",
    "radiance_to_bt",
    "radiance_to_bt_K",
    # Radiance ↔ reflectance
    "radiance_to_toa_reflectance",
    "radiance_sample_to_toa_reflectance",
    "toa_reflectance_to_radiance",
    "toa_reflectance_sample_to_radiance",
    "radiance_to_reflectance",
    "reflectance_to_radiance",
    "radiance_to_reflectance_avirisng",
    "reflectance_to_radiance_avirisng",
    "avirisng_bad_band_mask",
    # SWIR regimes
    "SWIRRegime",
    "classify_rt_regime",
    "classify_swir_regime",
    "swir_regime_for_sample",
    "swir_regime_for_scene",
    "attach_swir_regime",
    "trusted_swir",
    # Solar helpers
    "get_reference_esun",
    "project_esun_to_bands",
    "earth_sun_distance_au",
    "earth_sun_distance_for_sample",
    "esun_for_sample",
    # Resampling and virtual sensors
    "convolve_to_bands",
    "interpolate_to_centers",
    "generate_gaussian_srf",
    "SyntheticSensorConfig",
    "simulate_virtual_sensor",
    # Continuum removal
    "BandDefinition",
    "BandMetrics",
    "build_continuum",
    "compute_band_area",
    "compute_band_asymmetry",
    "compute_band_depth",
    "compute_band_metrics",
    "compute_composite_depth_ratio",
    "compute_convex_hull_continuum",
    "continuum_remove",
    # TES scaffolding
    "radiance_spectrum_to_bt_spectrum",
    "bt_spectrum_to_radiance_spectrum",
    "compute_lwir_emissivity_proxy",
    "lwir_pipeline_for_sample",
    "tes_lwirt",
    # Augmentation helpers
    "augment_radiance",
    "random_swirlike_atmosphere",
    # Misc
    "band_depth",
]
