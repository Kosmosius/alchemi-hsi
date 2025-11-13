import numpy as np
from sklearn.metrics import average_precision_score


def spectral_angle(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    num = float(np.dot(x, y))
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
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
