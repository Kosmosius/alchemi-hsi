import numpy as np

from alchemi.training.metrics import spectral_angle


def test_sam_zero_on_identical():
    x = np.array([1, 2, 3], dtype=float)
    assert spectral_angle(x, x) < 1e-8
