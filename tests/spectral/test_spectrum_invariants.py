import numpy as np
import pytest

from alchemi.spectral import Spectrum


def test_wavelength_monotonicity_and_ranges():
    with pytest.raises(ValueError):
        Spectrum(wavelength_nm=np.array([500.0, 400.0]), values=np.ones(2), kind="radiance")

    with pytest.raises(ValueError):
        Spectrum(wavelength_nm=np.array([500.0, 500.0]), values=np.ones(2), kind="radiance")

    spectrum = Spectrum(wavelength_nm=np.array([400.0, 500.0, 600.0]), values=np.array([0.1, 0.2, 0.3]), kind="radiance")
    np.testing.assert_array_equal(spectrum.wavelength_nm, np.array([400.0, 500.0, 600.0]))


@pytest.mark.parametrize("factory, expected_kind, scale", [
    (lambda: Spectrum.from_microns(np.array([0.4, 0.5]), np.array([1.0, 2.0]), kind="radiance"), "radiance", 1e-3),
    (lambda: Spectrum.from_wavenumber(np.array([25_000.0, 20_000.0]), np.array([4.0, 5.0]), kind="radiance"), "radiance", None),
    (lambda: Spectrum.from_microns(np.array([0.4, 0.5]), np.array([0.5, 0.6]), kind="reflectance"), "reflectance", None),
])
def test_unit_conversions_are_sorted(factory, expected_kind, scale):
    spectrum = factory()
    assert spectrum.kind == expected_kind
    assert np.all(np.diff(spectrum.wavelength_nm) > 0)
    if scale is not None:
        assert np.isclose(spectrum.values[0], scale, rtol=1e-12)


@pytest.mark.parametrize("values, should_raise", [
    (np.array([0.0, 0.5, 1.0]), False),
    (np.array([-0.5, 0.2, 0.3]), True),
    (np.array([0.1, 2.5, 0.2]), True),
])
def test_reflectance_ranges(values, should_raise):
    args = dict(wavelength_nm=np.array([400.0, 500.0, 600.0]), values=values, kind="reflectance")
    if should_raise:
        with pytest.raises(ValueError):
            Spectrum(**args)
    else:
        spec = Spectrum(**args)
        np.testing.assert_array_equal(spec.values, values)
