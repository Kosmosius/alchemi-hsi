import numpy as np

from alchemi.srf import emit_srf_matrix
from alchemi.srf.batch_convolve import batch_convolve_lab_to_sensor


def _highres_grid() -> np.ndarray:
    return np.linspace(380.0, 2500.0, 2000)


def test_srf_integral_emit():
    highres = _highres_grid()
    srf = emit_srf_matrix(highres)
    integrals = srf.row_integrals()
    assert np.allclose(integrals, 1.0, atol=1e-6)


def test_srf_flat_response_emit():
    highres = _highres_grid()
    srf = emit_srf_matrix(highres)
    flat = np.ones((1, highres.size), dtype=np.float64)
    convolved = batch_convolve_lab_to_sensor(highres, flat, srf)
    assert convolved.shape == (1, len(srf.centers_nm))
    assert np.allclose(convolved, 1.0, atol=1e-6)


def test_srf_cache_key_emit():
    highres = _highres_grid()
    srf = emit_srf_matrix(highres)
    assert srf.cache_key is not None
    assert srf.cache_key.startswith("emit:v01:")
    assert len(srf.cache_key.split(":")) == 3
