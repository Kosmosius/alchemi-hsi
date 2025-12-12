"""Micro-benchmarks for core physics operations.

This script is intentionally lightweight so it can be run locally to sanity-check
performance when adjusting Planck routines, resampling, continuum metrics, or
radiance/reflectance conversions. It avoids external dependencies beyond NumPy
and the ALCHEMI codebase.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable

import numpy as np

from alchemi.physics.planck import planck_radiance_wavelength, radiance_to_bt_K
from alchemi.physics.resampling import convolve_to_bands_batched, generate_gaussian_srf
from alchemi.physics.continuum import compute_band_metrics
from alchemi.physics.rad_reflectance import radiance_to_toa_reflectance
from alchemi.spectral.srf import SRFMatrix
from alchemi.types import BandDefinition, RadianceUnits, Spectrum, WavelengthGrid


def _time_call(name: str, func: Callable[[], object]) -> float:
    start = time.perf_counter()
    func()
    duration = time.perf_counter() - start
    print(f"{name:<35s}: {duration:6.3f} s")
    return duration


def _random_radiance_spectrum(n_pixels: int, n_bands: int, *, seed: int = 0) -> Spectrum:
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(400.0, 2500.0, n_bands, dtype=np.float64)
    values = rng.random((n_pixels, n_bands), dtype=np.float64)
    return Spectrum.from_radiance(
        WavelengthGrid(wavelengths),
        values,
        units=RadianceUnits.W_M2_SR_NM,
    )


def benchmark_planck(sizes: Iterable[tuple[int, int]]) -> None:
    print("\nPlanck radiance / inversion")
    for pixels, bands in sizes:
        spec = _random_radiance_spectrum(pixels, bands, seed=1)
        wavelengths = spec.wavelengths.nm
        temps = np.full((pixels, bands), 300.0, dtype=np.float64)

        _time_call(
            f"planck_radiance_wavelength {pixels}x{bands}",
            lambda: planck_radiance_wavelength(wavelengths, temps),
        )
        _time_call(
            f"radiance_to_bt_K          {pixels}x{bands}",
            lambda: radiance_to_bt_K(spec.values, wavelengths),
        )


def benchmark_resampling(sizes: Iterable[tuple[int, int]]) -> None:
    print("\nSRF convolution")
    srf = generate_gaussian_srf("benchmark", (400.0, 2500.0), 16)
    srf_dense = SRFMatrix(
        srf.sensor_name,
        srf.centers_nm,
        [srf.wavelength_nm] * len(srf.centers_nm),
        [row for row in srf.matrix],
    )

    for pixels, bands in sizes:
        spec = _random_radiance_spectrum(pixels, bands, seed=2)
        _time_call(
            f"convolve_to_bands_batched {pixels}x{bands}",
            lambda: convolve_to_bands_batched(spec.values, spec.wavelengths.nm, srf_dense),
        )


def benchmark_continuum(sizes: Iterable[tuple[int, int]]) -> None:
    print("\nContinuum and band metrics")
    band = BandDefinition(lambda_center_nm=950.0, lambda_left_nm=900.0, lambda_right_nm=1000.0)
    for pixels, bands in sizes:
        rng = np.random.default_rng(3)
        wavelengths = np.linspace(800.0, 1100.0, bands)
        reflectance = rng.random((pixels, bands))
        spectrum = Spectrum.from_reflectance(WavelengthGrid(wavelengths), reflectance)
        _time_call(
            f"compute_band_metrics      {pixels}x{bands}",
            lambda: compute_band_metrics(spectrum, band=band),
        )


def benchmark_radiance_reflectance(sizes: Iterable[tuple[int, int]]) -> None:
    print("\nRadiance ↔ reflectance")
    esun = np.linspace(1500.0, 1600.0, sizes[-1][1])
    for pixels, bands in sizes:
        spectrum = _random_radiance_spectrum(pixels, bands, seed=4)
        _time_call(
            f"radiance_to_toa_reflectance {pixels}x{bands}",
            lambda: radiance_to_toa_reflectance(
                spectrum, esun_band=esun[:bands], d_au=1.0, solar_zenith_deg=30.0
            ),
        )


def main() -> None:  # pragma: no cover - diagnostic script
    sizes = (
        (16, 64),
        (10_000, 200),
        (100_000, 200),
    )

    benchmark_planck(sizes)
    benchmark_resampling(sizes)
    benchmark_continuum(sizes)
    benchmark_radiance_reflectance(sizes)


if __name__ == "__main__":  # pragma: no cover - diagnostic script
    main()
