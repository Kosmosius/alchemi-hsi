from __future__ import annotations

import numpy as np

from alchemi.physics.swir import radiance_to_reflectance, reflectance_to_radiance


def test_swir_radiance_reflectance_round_trip():
    wavelengths = np.linspace(1000.0, 2500.0, 8)
    reflectance = np.full_like(wavelengths, 0.5, dtype=np.float64)

    E0 = np.linspace(1650.0, 1425.0, reflectance.size)  # solar irradiance [W/m^2/um]
    cos_sun = 0.85
    tau = 0.9
    Lpath = 0.1

    radiance = reflectance_to_radiance(reflectance, E0, cos_sun, tau, Lpath)
    recovered = radiance_to_reflectance(radiance, E0, cos_sun, tau, Lpath)

    np.testing.assert_allclose(recovered, reflectance, rtol=1e-6, atol=1e-4)


def test_swir_radiance_reflectance_variations():
    base_reflectance = np.array([0.1, 0.25, 0.4, 0.6, 0.8], dtype=np.float64)
    E0 = np.array([1500.0, 1600.0, 1700.0, 1750.0, 1800.0], dtype=np.float64)

    configs = [
        (0.75, 0.05),
        (0.9, 0.0),
        (1.0, 0.2),
    ]

    for cos_sun, Lpath in configs:
        radiance = reflectance_to_radiance(base_reflectance, E0, cos_sun, tau=0.92, Lpath=Lpath)
        recovered = radiance_to_reflectance(radiance, E0, cos_sun, tau=0.92, Lpath=Lpath)
        np.testing.assert_allclose(recovered, base_reflectance, rtol=1e-6, atol=5e-4)
