import numpy as np

from alchemi.physics import bt_to_radiance, radiance_to_bt


def test_bt_roundtrip():
    nm = np.array([10000.0], dtype=np.float64)
    T = np.array([300.0], dtype=np.float64)
    L = bt_to_radiance(T, nm)
    T2 = radiance_to_bt(L, nm)
    assert abs(T2[0] - 300.0) < 0.2
