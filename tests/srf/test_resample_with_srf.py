import numpy as np

from alchemi.srf.resample import resample_values_with_srf, resample_with_srf
from alchemi.srf.sensor import SensorSRF
from alchemi.types import QuantityKind, RadianceUnits, Spectrum, WavelengthGrid


def _rectangular_srf() -> SensorSRF:
    wl = np.linspace(400.0, 470.0, 8, dtype=np.float64)
    srfs = np.array(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=np.float64,
    )
    centers = np.array([410.0, 460.0], dtype=np.float64)
    widths = np.array([20.0, 20.0], dtype=np.float64)
    return SensorSRF(
        wavelength_grid_nm=wl, srfs=srfs, band_centers_nm=centers, band_widths_nm=widths
    )


def test_resample_with_rectangular_srf_single_and_batch():
    sensor_srf = _rectangular_srf()
    wavelengths = WavelengthGrid(sensor_srf.wavelength_grid_nm)
    values = np.arange(8, dtype=np.float64)  # 0..7
    spectrum = Spectrum(
        wavelengths=wavelengths,
        values=values,
        kind=QuantityKind.RADIANCE,
        units=RadianceUnits.W_M2_SR_NM,
    )

    resampled = resample_with_srf(spectrum, sensor_srf)
    # Expected averages over the first/second halves
    assert np.allclose(resampled.values, [1.5, 5.5])

    batched_values = np.vstack([values, values + 1.0])
    batched = Spectrum(
        wavelengths=wavelengths,
        values=batched_values,
        kind=QuantityKind.RADIANCE,
        units=RadianceUnits.W_M2_SR_NM,
    )
    resampled_batch = resample_with_srf(batched, sensor_srf)
    assert resampled_batch.values.shape == (2, 2)
    np.testing.assert_allclose(resampled_batch.values[0], resampled.values)
    np.testing.assert_allclose(resampled_batch.values[1], resampled.values + 1.0)


def test_flat_in_flat_out():
    sensor_srf = _rectangular_srf()
    wavelengths = WavelengthGrid(sensor_srf.wavelength_grid_nm)
    flat_values = np.ones_like(sensor_srf.wavelength_grid_nm) * 3.14
    spectrum = Spectrum(
        wavelengths=wavelengths,
        values=flat_values,
        kind=QuantityKind.RADIANCE,
        units=RadianceUnits.W_M2_SR_NM,
    )

    resampled = resample_with_srf(spectrum, sensor_srf)
    np.testing.assert_allclose(resampled.values, flat_values[0], atol=1e-6)


def test_grid_mismatch_triggers_interpolation():
    sensor_srf = _rectangular_srf()
    # Slightly perturbed grid
    hr_grid = sensor_srf.wavelength_grid_nm + 5e-4
    values = np.linspace(1.0, 2.0, hr_grid.size, dtype=np.float64)
    spectrum = Spectrum(
        wavelengths=WavelengthGrid(hr_grid),
        values=values,
        kind=QuantityKind.RADIANCE,
        units=RadianceUnits.W_M2_SR_NM,
    )

    resampled = resample_with_srf(spectrum, sensor_srf, allow_mismatch_tol_nm=1e-5)
    expected, _ = resample_values_with_srf(values, hr_grid, sensor_srf, allow_mismatch_tol_nm=1e-5)
    np.testing.assert_allclose(resampled.values, expected)
