"""Confidence calibration utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .metrics import ece_score


class TemperatureScaler:
    """Platt-style temperature scaling for logits."""

    def __init__(self) -> None:
        self.T: float = 1.0

    def fit(self, logits: NDArray[np.floating[Any]], labels: NDArray[np.integer[Any]]) -> None:
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)

        if logits.ndim == 1 or logits.shape[-1] == 1:
            predictions = (logits >= 0.0).astype(int)

            def confidences(arr: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
                arr = np.asarray(arr, dtype=np.float64)
                probs = 1.0 / (1.0 + np.exp(-arr))
                return np.maximum(probs, 1.0 - probs).astype(np.float64, copy=False)

        else:
            predictions = logits.argmax(axis=1)

            def confidences(arr: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
                arr = np.asarray(arr, dtype=np.float64)
                preds = np.argmax(arr, axis=1)
                top = arr[np.arange(arr.shape[0]), preds]
                return np.exp(top - logsumexp(arr, axis=1)).astype(np.float64, copy=False)

        correct = (predictions == labels).astype(np.float64, copy=False)

        def objective(temperature: float) -> float:
            scaled = logits / max(temperature, 1e-6)
            conf = confidences(scaled)
            return ece_score(conf.astype(float), correct)

        baseline = objective(1.0)
        result = minimize_scalar(objective, bounds=(0.5, 5.0), method="bounded")
        candidate = float(result.x) if result.success else 1.0
        self.T = candidate if objective(candidate) <= baseline else 1.0

    def transform(self, logits: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        arr = np.asarray(logits, dtype=np.float64)
        return arr / max(self.T, 1e-6)


def _nll(logits: NDArray[np.floating[Any]], labels: NDArray[np.integer[Any]]) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if logits.ndim == 1 or logits.shape[-1] == 1:
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        probabilities = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
        return float(
            -np.mean(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities))
        )

    log_partition = logsumexp(logits, axis=-1)
    log_likelihood = logits[np.arange(len(labels)), labels] - log_partition
    return float(-np.mean(log_likelihood))
