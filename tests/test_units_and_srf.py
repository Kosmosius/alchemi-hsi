import warnings

import numpy as np
import pytest

from alchemi.spectra.data.spectralearth import load_spectralearth
from alchemi.srf.convolve import convolve_lab_to_sensor
from alchemi.types import Spectrum, SpectrumKind, SRFMatrix, WavelengthGrid


def test_wavelength_conversion_and_validation():
    wavelengths_um = np.array([0.4, 0.5, 0.6])
    reflectance = np.full_like(wavelengths_um, 0.25)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spectrum = load_spectralearth(
            {
                "wavelengths": wavelengths_um,
                "wavelength_unit": "micrometers",
                "reflectance": reflectance,
            }
        )

    assert np.allclose(spectrum.wavelengths.nm, wavelengths_um * 1000.0)
    assert spectrum.kind is SpectrumKind.REFLECTANCE
    assert spectrum.units == "1"
    assert any("micrometre" in str(w.message) for w in caught)

    with pytest.raises(ValueError):
        load_spectralearth(
            {"wavelengths": [0.6, 0.5, 0.4], "wavelength_unit": "micron", "radiance": [1, 2, 3]}
        )


def test_band_mask_propagation():
    sample = {
        "wavelength_nm": [400.0, 410.0, 420.0],
        "radiance": [1.0, 2.0, 3.0],
        "radiance_units": "W·m⁻²·sr⁻¹·nm⁻¹",
        "band_valid_mask": [True, False, True],
    }

    spectrum = load_spectralearth(sample)

    assert spectrum.mask is not None
    assert spectrum.mask.dtype == bool
    assert spectrum.mask.tolist() == [True, False, True]


def test_flat_spectrum_remains_flat_with_normalized_srf():
    wavelength_grid = np.linspace(400.0, 700.0, 301)
    flat = Spectrum(
        WavelengthGrid(wavelength_grid),
        np.full_like(wavelength_grid, 0.3),
        SpectrumKind.REFLECTANCE,
        "1",
    )

    centers = np.array([450.0, 550.0, 650.0])
    bands_nm = [
        np.array([430.0, 440.0, 450.0, 460.0, 470.0]),
        np.array([520.0, 540.0, 560.0]),
        np.array([630.0, 640.0, 650.0, 660.0]),
    ]
    bands_resp = [
        np.array([0.0, 0.5, 1.0, 0.5, 0.0]),
        np.array([0.1, 0.8, 0.1]),
        np.array([0.0, 1.0, 0.0, 0.0]),
    ]

    srf = SRFMatrix("mock", centers, bands_nm, bands_resp).normalize_trapz()
    convolved = convolve_lab_to_sensor(flat, srf)

    assert convolved.values.shape == centers.shape
    assert np.allclose(convolved.values, 0.3, atol=1e-6)
    assert convolved.kind is SpectrumKind.REFLECTANCE
