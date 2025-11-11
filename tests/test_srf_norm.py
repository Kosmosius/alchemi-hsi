import numpy as np

from alchemi.types import SRFMatrix


def test_trapz_norm():
    centers = np.array([1000.0, 1100.0])
    bands_nm = [np.array([990, 1000, 1010]), np.array([1090, 1100, 1110])]
    bands_resp = [np.array([0, 1, 0], dtype=float), np.array([0, 1, 0], dtype=float)]
    srf = SRFMatrix("test", centers, bands_nm, bands_resp).normalize_trapz()
    ints = srf.row_integrals()
    assert np.allclose(ints, 1.0, atol=1e-6)
