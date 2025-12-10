import numpy as np
import pytest

from alchemi.spectral import Spectrum


def test_wavelength_monotonicity_and_ranges():
    with pytest.raises(ValueError):
        Spectrum(wavelength_nm=np.array([500.0, 400.0]), values=np.ones(2), kind="radiance")

    with pytest.raises(ValueError):
        Spectrum(wavelength_nm=np.array([500.0, 500.0]), values=np.ones(2), kind="radiance")

    spectrum = Spectrum(
        wavelength_nm=np.array([400.0, 500.0, 600.0]),
        values=np.array([0.1, 0.2, 0.3]),
        kind="radiance",
    )
    np.testing.assert_array_equal(spectrum.wavelength_nm, np.array([400.0, 500.0, 600.0]))


@pytest.mark.parametrize(
    "factory, expected_kind, scale",
    [
        (
            lambda: Spectrum.from_microns(
                np.array([0.4, 0.5]), np.array([1.0, 2.0]), kind="radiance"
            ),
            "radiance",
            1e-3,
        ),
        (
            lambda: Spectrum.from_wavenumber(
                np.array([25_000.0, 20_000.0]), np.array([4.0, 5.0]), kind="radiance"
            ),
            "radiance",
            None,
        ),
        (
            lambda: Spectrum.from_microns(
                np.array([0.4, 0.5]), np.array([0.5, 0.6]), kind="reflectance"
            ),
            "reflectance",
            None,
        ),
    ],
)
def test_unit_conversions_are_sorted(factory, expected_kind, scale):
    spectrum = factory()
    assert spectrum.kind == expected_kind
    assert np.all(np.diff(spectrum.wavelength_nm) > 0)
    if scale is not None:
        assert np.isclose(spectrum.values[0], scale, rtol=1e-12)


def planck_radiance_wavenumber_cm(wavenumber_cm: np.ndarray, temperature_K: float) -> np.ndarray:
    """Planck's law expressed per wavenumber in ``cm⁻¹``."""

    h = 6.626_070_15e-34
    c = 2.997_924_58e8
    k_b = 1.380_649e-23

    sigma_m = np.asarray(wavenumber_cm, dtype=np.float64) * 100.0
    exponent = (h * c * sigma_m) / (k_b * temperature_K)
    exponent = np.clip(exponent, 0.0, 700.0)
    prefactor = 2.0 * h * c**2 * sigma_m**3

    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        expm1_term = np.expm1(exponent)

    tiny_exponent = exponent < 1e-6
    if np.any(tiny_exponent):
        expm1_term = np.where(
            tiny_exponent,
            exponent + 0.5 * exponent**2,
            expm1_term,
        )

    radiance_per_m = prefactor / expm1_term
    return radiance_per_m * 100.0


def test_wavenumber_conversion_matches_planck():
    temperature_K = 300.0
    wavenumber_cm = np.array([1300.0, 1000.0, 800.0])

    radiance_per_cm = planck_radiance_wavenumber_cm(wavenumber_cm, temperature_K)
    spectrum = Spectrum.from_wavenumber(wavenumber_cm, radiance_per_cm, kind="radiance")

    expected_wavelengths_nm = 1.0e7 / wavenumber_cm
    np.testing.assert_allclose(spectrum.wavelength_nm, np.sort(expected_wavelengths_nm))

    from alchemi.physics.planck import planck_radiance_wavelength

    expected_radiance = planck_radiance_wavelength(spectrum.wavelength_nm, temperature_K)
    np.testing.assert_allclose(spectrum.values, expected_radiance, rtol=1e-10, atol=0.0)


@pytest.mark.parametrize(
    "values, should_raise",
    [
        (np.array([0.0, 0.5, 1.0]), False),
        (np.array([-0.5, 0.2, 0.3]), True),
        (np.array([0.1, 2.5, 0.2]), True),
    ],
)
def test_reflectance_ranges(values, should_raise):
    args = dict(
        wavelength_nm=np.array([400.0, 500.0, 600.0]),
        values=values,
        kind="reflectance",
        units="fraction",
    )
    if should_raise:
        with pytest.raises(ValueError):
            Spectrum(**args)
    else:
        spec = Spectrum(**args)
        np.testing.assert_array_equal(spec.values, values)
