import numpy as np
import pytest

from alchemi.physics.resampling import (
    convolve_to_bands,
    convolve_to_bands_batched,
    generate_gaussian_srf,
    interpolate_to_centers,
)
from alchemi.spectral.srf import SRFMatrix as DenseSRFMatrix
from alchemi.types import Spectrum, WavelengthGrid


def test_flat_spectrum_remains_flat_when_resampled():
    wavelengths = WavelengthGrid(np.linspace(400.0, 500.0, 200))
    flat_values = np.full_like(wavelengths.nm, fill_value=7.5)
    spectrum = Spectrum.from_radiance(wavelengths, flat_values)
    srf = generate_gaussian_srf("toy", (400.0, 500.0), num_bands=9, fwhm_nm=8.0)
    banded = convolve_to_bands(spectrum, srf)
    assert np.allclose(banded.values, banded.values[0], atol=1e-6)


def test_gaussian_line_survives_interpolation():
    wavelengths = WavelengthGrid(np.linspace(750.0, 850.0, 101))
    center = 800.0
    sigma = 5.0
    values = np.exp(-0.5 * ((wavelengths.nm - center) / sigma) ** 2)
    spectrum = Spectrum.from_radiance(wavelengths, values)
    centers = np.array([790.0, 800.0, 810.0])
    interpolated = interpolate_to_centers(spectrum, centers, mode="linear")
    peak_idx = np.argmax(interpolated.values)
    assert centers[peak_idx] == 800.0
    assert interpolated.values[peak_idx] > interpolated.values.mean()


def test_gaussian_line_survives_srf_convolution():
    wavelengths = WavelengthGrid(np.linspace(750.0, 850.0, 2001))
    center = 802.0
    sigma = 0.7
    values = np.exp(-0.5 * ((wavelengths.nm - center) / sigma) ** 2)
    spectrum = Spectrum.from_radiance(wavelengths, values)

    srf = generate_gaussian_srf("toy", (750.0, 850.0), num_bands=11, fwhm_nm=10.0)
    convolved = convolve_to_bands(spectrum, srf)

    peak_idx = int(np.argmax(convolved.values))
    closest_idx = int(np.argmin(np.abs(srf.centers_nm - center)))
    assert peak_idx == closest_idx
    assert convolved.values[peak_idx] > 0.1
    assert convolved.values[peak_idx] >= convolved.values.mean()


def test_linear_interpolation_matches_monotonic_line():
    source_wavelengths = np.linspace(400.0, 500.0, 11)
    slope = 2.5
    intercept = -17.0
    source_values = slope * source_wavelengths + intercept
    target_centers = np.linspace(405.0, 495.0, 19)

    interpolated = interpolate_to_centers(
        source_wavelengths, source_values, target_centers, mode="linear"
    )
    expected = slope * target_centers + intercept

    assert np.allclose(interpolated, expected, atol=1e-9)


def test_nearest_neighbor_mode_matches_closest_source():
    source_wavelengths = np.array([0.0, 1.0, 2.0, 3.0])
    source_values = np.array([0.0, 10.0, 20.0, 30.0])
    target_centers = np.array([0.1, 0.9, 1.6, 2.6, 3.4])

    interpolated = interpolate_to_centers(
        source_wavelengths, source_values, target_centers, mode="nearest"
    )

    assert np.allclose(interpolated[:-1], np.array([0.0, 10.0, 20.0, 30.0]))
    assert np.isnan(interpolated[-1])


def test_interpolation_vectorizes_over_leading_dimensions():
    source_wavelengths = np.array([500.0, 505.0, 510.0])
    source_values = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    target_centers = np.array([500.0, 507.5, 510.0])

    interpolated = interpolate_to_centers(
        source_wavelengths, source_values, target_centers, mode="linear"
    )

    assert interpolated.shape == (2, 3)
    assert np.allclose(interpolated[0], np.array([1.0, 2.5, 3.0]))
    assert np.allclose(interpolated[1], np.array([10.0, 25.0, 30.0]))


def test_spline_interpolation_outperforms_linear_for_smooth_signal():
    pytest.importorskip("scipy")

    source_wavelengths = np.linspace(0.0, 2 * np.pi, 12)
    source_values = np.sin(source_wavelengths)
    target_centers = np.linspace(0.0, 2 * np.pi, 101)
    truth = np.sin(target_centers)

    linear = interpolate_to_centers(
        source_wavelengths,
        source_values,
        target_centers,
        mode="linear",
        allow_extrapolation=True,
    )
    spline = interpolate_to_centers(
        source_wavelengths,
        source_values,
        target_centers,
        mode="spline",
        allow_extrapolation=True,
    )

    linear_error = np.linalg.norm(linear - truth)
    spline_error = np.linalg.norm(spline - truth)

    assert spline_error < linear_error
    assert np.isfinite(spline_error)


def test_batched_convolution_matches_per_band_path():
    wavelengths = WavelengthGrid(np.linspace(400.0, 500.0, 50))
    flat_values = np.ones_like(wavelengths.nm)
    spectrum = Spectrum.from_radiance(wavelengths, flat_values)
    srf = generate_gaussian_srf("toy", (400.0, 500.0), num_bands=4, fwhm_nm=5.0)

    dense_matrix = []
    for nm_band, resp in zip(srf.bands_nm, srf.bands_resp, strict=True):
        band_resp = np.interp(wavelengths.nm, nm_band, resp, left=0.0, right=0.0)
        band_resp /= np.trapz(band_resp, x=wavelengths.nm)
        dense_matrix.append(band_resp)
    dense_srf = DenseSRFMatrix(wavelengths.nm, np.vstack(dense_matrix))

    single_batched = convolve_to_bands_batched(
        flat_values[np.newaxis, :], wavelengths.nm, dense_srf
    )[0]
    legacy = convolve_to_bands(spectrum, srf).values

    assert np.allclose(single_batched, legacy, atol=1e-6)


def test_batched_convolution_handles_multiple_spectra_and_shapes():
    wavelengths = WavelengthGrid(np.linspace(750.0, 850.0, 101))
    flat_values = np.ones_like(wavelengths.nm)
    center = 800.0
    sigma = 4.0
    gaussian_values = np.exp(-0.5 * ((wavelengths.nm - center) / sigma) ** 2)

    srf = generate_gaussian_srf("toy", (750.0, 850.0), num_bands=5, fwhm_nm=8.0)
    dense_matrix = []
    for nm_band, resp in zip(srf.bands_nm, srf.bands_resp, strict=True):
        band_resp = np.interp(wavelengths.nm, nm_band, resp, left=0.0, right=0.0)
        band_resp /= np.trapz(band_resp, x=wavelengths.nm)
        dense_matrix.append(band_resp)
    dense_srf = DenseSRFMatrix(wavelengths.nm, np.vstack(dense_matrix))

    batched = np.stack([flat_values, gaussian_values])
    reshaped = batched.reshape(1, 2, wavelengths.nm.size)
    band_values = convolve_to_bands_batched(reshaped, wavelengths.nm, dense_srf)

    assert band_values.shape == (1, 2, dense_srf.matrix.shape[0])

    flat_bands = band_values[0, 0]
    assert np.allclose(flat_bands, flat_bands[0], atol=1e-6)

    gaussian_bands = band_values[0, 1]
    peak_idx = np.argmax(gaussian_bands)
    closest_idx = int(np.argmin(np.abs(srf.centers_nm - center)))
    assert peak_idx == closest_idx
    assert gaussian_bands[peak_idx] >= gaussian_bands.mean()


def test_convolution_rejects_zero_area_rows():
    wavelengths = WavelengthGrid(np.linspace(400.0, 410.0, 5))
    flat_values = np.ones((2, wavelengths.nm.size), dtype=np.float64)
    srf_matrix = np.zeros((3, wavelengths.nm.size), dtype=np.float64)

    with pytest.raises(ValueError):
        convolve_to_bands_batched(flat_values, wavelengths.nm, srf_matrix)
