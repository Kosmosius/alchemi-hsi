import numpy as np

from alchemi.training.calibration import TemperatureScaler
from alchemi.training.metrics import ece_score


def test_temp_scaling_reduces_ece():
    logits = np.array([[2.0, 0.1], [0.1, 1.0], [1.2, 0.2], [0.2, 1.5]])
    labels = np.array([0, 1, 0, 1])
    conf = _predicted_confidences(logits)
    correct = (logits.argmax(axis=1) == labels).astype(float)
    e0 = ece_score(conf, correct)
    ts = TemperatureScaler()
    ts.fit(logits, labels)
    logits_c = ts.transform(logits)
    conf_c = _predicted_confidences(logits_c)
    e1 = ece_score(conf_c, correct)
    assert e1 <= e0 + 1e-6


def _predicted_confidences(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    probs = exps / np.maximum(exps.sum(axis=1, keepdims=True), 1e-12)
    return probs.max(axis=1).astype(float)
