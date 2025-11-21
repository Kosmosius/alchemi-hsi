import numpy as np
import pytest

from alchemi.types import SRFMatrix


def test_valid_srf_matrix() -> None:
    centers = np.array([500.0, 600.0])
    bands_nm = [np.array([490.0, 500.0, 510.0]), np.array([590.0, 600.0, 610.0])]
    bands_resp = [np.array([0.1, 0.5, 0.1]), np.array([0.2, 0.6, 0.2])]
    SRFMatrix("sensor", centers, bands_nm, bands_resp)


def test_length_mismatch_raises() -> None:
    centers = np.array([500.0])
    bands_nm = [np.array([490.0, 500.0, 510.0]), np.array([590.0, 600.0, 610.0])]
    bands_resp = [np.array([0.1, 0.5, 0.1]), np.array([0.2, 0.6, 0.2])]
    with pytest.raises(ValueError, match="centers_nm length must match"):
        SRFMatrix("sensor", centers, bands_nm, bands_resp)


def test_band_length_mismatch_raises() -> None:
    centers = np.array([500.0, 600.0])
    bands_nm = [np.array([490.0, 500.0, 510.0]), np.array([590.0, 600.0, 610.0])]
    bands_resp = [np.array([0.1, 0.5, 0.1]), np.array([0.2, 0.6])]
    with pytest.raises(ValueError, match="must have the same length"):
        SRFMatrix("sensor", centers, bands_nm, bands_resp)


def test_non_1d_inputs_raise() -> None:
    centers = np.array([500.0])
    bands_nm = [np.array([[490.0, 500.0, 510.0]])]
    bands_resp = [np.array([0.1, 0.5, 0.1])]
    with pytest.raises(ValueError, match="must be 1-D"):
        SRFMatrix("sensor", centers, bands_nm, bands_resp)

    bands_nm = [np.array([490.0, 500.0, 510.0])]
    bands_resp = [np.array([[0.1, 0.5, 0.1]])]
    with pytest.raises(ValueError, match="must be 1-D"):
        SRFMatrix("sensor", centers, bands_nm, bands_resp)
