import numpy as np


class TemperatureScaler:
    def __init__(self):
        self.T = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        Ts = np.linspace(0.5, 5.0, 60)
        best_T, best_nll = 1.0, 1e9
        for T in Ts:
            nll = _nll(logits / T, labels)
            if nll < best_nll:
                best_nll, best_T = nll, T
        self.T = float(best_T)

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return logits / self.T


def _nll(logits: np.ndarray, labels: np.ndarray) -> float:
    from scipy.special import logsumexp

    if logits.ndim == 1:
        p = 1.0 / (1.0 + np.exp(-logits))
        eps = 1e-8
        return -np.mean(labels * np.log(p + eps) + (1 - labels) * np.log(1 - p + eps))
    else:
        lse = logsumexp(logits, axis=-1)
        ll = logits[np.arange(len(labels)), labels] - lse
        return -float(np.mean(ll))
