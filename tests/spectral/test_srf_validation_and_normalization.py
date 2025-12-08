"""SRF validation tests.

Numerical tolerances are intentionally loose (rtol=1e-3, atol=1e-3) to account
for the coarse sampling of some SRFs while still catching unnormalized rows or
flat-spectrum violations.
"""

from pathlib import Path

import numpy as np
import pytest

from alchemi.registry import srfs
from alchemi.spectral import Spectrum, SRFMatrix


pytestmark = pytest.mark.physics_and_metadata

def test_row_normalization_preserves_flat_spectrum():
    wavelength_nm = np.linspace(400.0, 500.0, 5)
    matrix = np.vstack([
        np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
        np.array([1.0, 2.0, 3.0, 2.0, 1.0]),
    ])
    srf = SRFMatrix(wavelength_nm=wavelength_nm, matrix=matrix.copy())
    srf.normalize_rows_trapz()
    flat = Spectrum(wavelength_nm=wavelength_nm, values=np.ones_like(wavelength_nm), kind="radiance")
    srf.assert_flat_spectrum_preserved(flat, tol=1e-6)


def test_negative_entries_raise():
    wavelength_nm = np.array([1.0, 2.0])
    srf = SRFMatrix(wavelength_nm=wavelength_nm, matrix=np.array([[0.5, -0.1]]))
    with pytest.raises(ValueError):
        srf.assert_nonnegative()


def test_invalid_normalization_guardrails():
    wavelength_nm = np.array([400.0, 500.0, 600.0])
    matrix = np.zeros((1, 3))
    srf = SRFMatrix(wavelength_nm=wavelength_nm, matrix=matrix)
    with pytest.raises(ValueError):
        srf.normalize_rows_trapz()


def _registered_sensor_ids() -> list[str]:
    srf_root = Path("resources/srfs")
    sensors = {
        path.name.split("_srfs", maxsplit=1)[0]
        for path in srf_root.iterdir()
        if path.is_file() and path.name.endswith(('_srfs.json', '_srfs.npy', '_srfs.npz'))
    }
    return sorted(sensors)


@pytest.mark.parametrize("sensor_id", _registered_sensor_ids())
def test_registered_srfs_have_unit_row_integrals(sensor_id: str):
    srf = srfs.get_srf(sensor_id)
    bad_band_mask = srf.bad_band_mask if srf.bad_band_mask is not None else np.zeros(len(srf.bands_resp), dtype=bool)

    for idx, (nm, resp, is_bad) in enumerate(zip(srf.bands_nm, srf.bands_resp, bad_band_mask, strict=True)):
        if is_bad:
            continue
        area = np.trapz(resp, x=nm)
        np.testing.assert_allclose(
            area,
            1.0,
            rtol=1e-3,
            atol=1e-3,
            err_msg=f"Band {idx} for sensor {sensor_id} does not integrate to 1",
        )


@pytest.mark.parametrize("sensor_id", _registered_sensor_ids())
def test_flat_spectrum_preserved_by_registered_srfs(sensor_id: str):
    srf = srfs.get_srf(sensor_id)
    bad_band_mask = srf.bad_band_mask if srf.bad_band_mask is not None else np.zeros(len(srf.bands_resp), dtype=bool)

    for idx, (nm, resp, is_bad) in enumerate(zip(srf.bands_nm, srf.bands_resp, bad_band_mask, strict=True)):
        if is_bad:
            continue
        flat_spectrum = np.ones_like(nm)
        band_value = np.trapz(resp * flat_spectrum, x=nm)
        np.testing.assert_allclose(
            band_value,
            1.0,
            rtol=1e-3,
            atol=1e-3,
            err_msg=f"Flat spectrum not preserved for band {idx} of sensor {sensor_id}",
        )
