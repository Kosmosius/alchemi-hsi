import numpy as np

from alchemi.physics.resampling import convolve_to_bands, generate_gaussian_srf, interpolate_to_centers
from alchemi.types import Spectrum, WavelengthGrid


def test_flat_spectrum_remains_flat_when_resampled():
    wavelengths = WavelengthGrid(np.linspace(400.0, 500.0, 50))
    flat_values = np.ones_like(wavelengths.nm)
    spectrum = Spectrum.from_radiance(wavelengths, flat_values)
    srf = generate_gaussian_srf("toy", (400.0, 500.0), num_bands=5, fwhm_nm=5.0)
    banded = convolve_to_bands(spectrum, srf)
    assert np.allclose(banded.values, banded.values[0], atol=1e-3)


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
