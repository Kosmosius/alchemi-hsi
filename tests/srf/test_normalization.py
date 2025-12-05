import numpy as np
import pytest

from alchemi.srf.utils import (
    check_flat_spectrum_invariant,
    normalize_srf_rows,
    validate_srf,
)


def _synthetic_srfs(wavelength_nm: np.ndarray) -> np.ndarray:
    tophat = np.where((wavelength_nm >= 2.0) & (wavelength_nm <= 4.0), 1.0, 0.2)
    triangle = np.maximum(0.0, 1.0 - np.abs(wavelength_nm - 3.0))
    ramp = np.linspace(0.1, 1.1, wavelength_nm.shape[0], dtype=np.float64)
    return np.vstack([tophat, triangle, ramp])


def test_normalize_and_validate_srfs() -> None:
    wl = np.linspace(0.0, 6.0, 13, dtype=np.float64)
    srfs = _synthetic_srfs(wl)

    normalized = normalize_srf_rows(wl, srfs)
    validate_srf(wl, normalized, allow_negative_eps=1e-9)
    check_flat_spectrum_invariant(wl, normalized, c=2.5, tol=1e-3)


def test_validate_rejects_zero_area() -> None:
    wl = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    srfs = np.zeros((2, wl.size), dtype=np.float64)

    with pytest.raises(ValueError):
        normalize_srf_rows(wl, srfs)

    normalized = srfs.copy()
    with pytest.raises(ValueError):
        validate_srf(wl, normalized, min_area=1e-6)


def test_validate_rejects_negative_values() -> None:
    wl = np.linspace(0.0, 2.0, 5, dtype=np.float64)
    srfs = np.full((1, wl.size), 0.2, dtype=np.float64)
    srfs[0, 2] = -0.5

    normalized = normalize_srf_rows(wl, srfs, atol=1e-6)
    with pytest.raises(ValueError):
        validate_srf(wl, normalized, allow_negative_eps=1e-3)


def test_validate_rejects_misnormalized_rows() -> None:
    wl = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    srfs = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)

    with pytest.raises(ValueError):
        validate_srf(wl, srfs, area_tol=1e-3)
