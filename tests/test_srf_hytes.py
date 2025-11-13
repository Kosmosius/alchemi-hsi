import numpy as np

from alchemi_hsi.srf.hytes import hytes_srf_matrix


def test_srf_integral_hytes():
    srf = hytes_srf_matrix()
    integrals = srf.row_integrals()
    assert np.allclose(integrals, 1.0, atol=1e-6)


def test_srf_flat_response_hytes():
    srf = hytes_srf_matrix()
    flat_value = 3.0
    outputs = [
        np.trapezoid(np.full_like(resp, flat_value) * resp, nm)
        for nm, resp in zip(srf.bands_nm, srf.bands_resp)
    ]
    outputs = np.asarray(outputs)
    assert np.allclose(outputs, flat_value, atol=1e-6)


def test_srf_cache_roundtrip_hytes():
    srf1 = hytes_srf_matrix()
    srf2 = hytes_srf_matrix()
    assert srf1 is srf2
    assert isinstance(srf1.cache_key, str)
    assert srf1.cache_key == srf2.cache_key
