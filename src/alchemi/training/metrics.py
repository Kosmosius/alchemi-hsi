import numpy as np
from sklearn.metrics import average_precision_score


def spectral_angle(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the spectral angle mapper (SAM) between two spectra.

    Invalid bands (NaN or inf) are ignored, mirroring common masked-band
    handling in hyperspectral processing. If no valid bands remain or either
    spectrum is empty, the function returns 0.0.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return 0.0

    x_valid = x[valid]
    y_valid = y[valid]
    if x_valid.size == 0 or y_valid.size == 0:
        return 0.0

    num = float(np.dot(x_valid, y_valid))
    den = float(np.linalg.norm(x_valid) * np.linalg.norm(y_valid))
    if den <= 0.0:
        return 0.0
    cosine = np.clip(num / den, -1.0, 1.0)
    return float(np.arccos(cosine))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))


def ece_score(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        sel = (conf >= bins[i]) & (conf < bins[i + 1])
        if sel.any():
            acc = correct[sel].mean()
            conf_m = conf[sel].mean()
            ece += abs(acc - conf_m) * sel.mean()
    return float(ece)
