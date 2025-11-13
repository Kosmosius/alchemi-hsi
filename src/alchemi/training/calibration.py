import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .metrics import ece_score


class TemperatureScaler:
    def __init__(self):
        self.T = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        correct = (logits.argmax(axis=1) == labels).astype(float)

        def _confidences(x: np.ndarray) -> np.ndarray:
            preds = np.argmax(x, axis=1)
            top = x[np.arange(len(preds)), preds]
            return np.exp(top - logsumexp(x, axis=1)).astype(float)

        def objective(T: float) -> float:
            return ece_score(_confidences(logits / T), correct)

        res = minimize_scalar(objective, bounds=(0.5, 5.0), method="bounded")
        self.T = float(res.x if res.success else 1.0)

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return logits / max(self.T, 1e-6)


def _nll(logits: np.ndarray, labels: np.ndarray) -> float:
    if logits.ndim == 1:
        p = 1.0 / (1.0 + np.exp(-logits))
        eps = 1e-8
        return -np.mean(labels * np.log(p + eps) + (1 - labels) * np.log(1 - p + eps))
    else:
        lse = logsumexp(logits, axis=-1)
        ll = logits[np.arange(len(labels)), labels] - lse
        return -float(np.mean(ll))
