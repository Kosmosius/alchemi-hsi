from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def ensure_1d(x: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def as_float32(x: ArrayLike) -> NDArray[np.float32]:
    return np.asarray(x, dtype=np.float32)
