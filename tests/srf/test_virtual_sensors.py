import numpy as np

from alchemi.srf.synthetic import make_virtual_sensor, perturb_sensor_srf
from alchemi.spectral.srf import SRFMatrix as DenseSRFMatrix
from alchemi.types import QuantityKind, Spectrum, ValueUnits


def _assert_normalized(srf_matrix: DenseSRFMatrix) -> None:
    integrals = np.trapz(srf_matrix.matrix, x=srf_matrix.wavelength_nm, axis=1)
    assert np.allclose(integrals, 1.0, atol=1e-6)
    flat = Spectrum(
        wavelength_nm=srf_matrix.wavelength_nm,
        values=np.ones_like(srf_matrix.wavelength_nm),
        kind=QuantityKind.REFLECTANCE,
        units=ValueUnits.REFLECTANCE_FRACTION,
    )
    srf_matrix.assert_flat_spectrum_preserved(flat, tol=1e-6)


def test_make_virtual_sensor_basic_generation():
    band_count = 5
    sensor = make_virtual_sensor(
        sensor_id="virtual-test",
        wavelength_min_nm=400.0,
        wavelength_max_nm=2500.0,
        band_count=band_count,
        base_fwhm_nm=15.0,
        center_jitter_nm=1.0,
        width_jitter_frac=0.1,
        grid_step_nm=2.0,
        rng=np.random.default_rng(42),
    )

    assert sensor.srfs.shape[0] == band_count
    assert sensor.srfs.shape[1] == sensor.wavelength_grid_nm.shape[0]
    assert np.all(sensor.band_centers_nm >= 400.0)
    assert np.all(sensor.band_centers_nm <= 2500.0)
    assert np.all(sensor.band_widths_nm > 0)
    assert sensor.provenance.name == "GAUSSIAN"

    dense = DenseSRFMatrix(sensor.wavelength_grid_nm, sensor.srfs)
    _assert_normalized(dense)


def test_perturb_sensor_srf_jitter_and_noise():
    base = make_virtual_sensor(
        wavelength_min_nm=800.0,
        wavelength_max_nm=820.0,
        band_count=3,
        base_fwhm_nm=2.0,
        rng=np.random.default_rng(7),
    )

    perturbed = perturb_sensor_srf(
        base,
        center_jitter_nm=0.5,
        width_jitter_frac=0.2,
        shape_noise_frac=0.05,
        rng=np.random.default_rng(11),
    )

    assert perturbed.srfs.shape == base.srfs.shape
    assert np.any(~np.isclose(perturbed.band_centers_nm, base.band_centers_nm))
    assert np.all(np.abs(perturbed.band_centers_nm - base.band_centers_nm) <= 0.5 + 1e-6)
    assert np.any(~np.isclose(perturbed.band_widths_nm, base.band_widths_nm))

    dense = DenseSRFMatrix(perturbed.wavelength_grid_nm, perturbed.srfs)
    _assert_normalized(dense)
    assert perturbed.provenance == base.provenance
