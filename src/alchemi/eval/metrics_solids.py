import numpy as np
from sklearn.metrics import f1_score

from ..physics.swir import band_depth


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def banddepth_mae(
    nm: np.ndarray,
    R_true: np.ndarray,
    R_pred: np.ndarray,
    windows: list[tuple[float, float, float]],
) -> float:
    errs = []
    for center, left, right in windows:
        d_t = band_depth(nm, R_true, center, left, right)
        d_p = band_depth(nm, R_pred, center, left, right)
        errs.append(abs(d_t - d_p))
    return float(np.mean(errs))
