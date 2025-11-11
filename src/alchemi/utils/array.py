import numpy as np


def ensure_1d(x):
    return np.asarray(x).reshape(-1)


def as_float32(x):
    return np.asarray(x, dtype=np.float32)
