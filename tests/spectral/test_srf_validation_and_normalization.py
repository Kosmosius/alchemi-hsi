import numpy as np
import pytest

from alchemi.spectral import Spectrum, SRFMatrix


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
