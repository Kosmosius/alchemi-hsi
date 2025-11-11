import numpy as np

from alchemi.training.calibration import TemperatureScaler
from alchemi.training.metrics import ece_score


def test_temp_scaling_reduces_ece():
    logits = np.array([[2.0, 0.1], [0.1, 1.0], [1.2, 0.2], [0.2, 1.5]])
    labels = np.array([0, 1, 0, 1])
    conf = (logits.max(axis=1) / np.exp(logits).sum(axis=1)).astype(float)
    correct = (logits.argmax(axis=1) == labels).astype(float)
    e0 = ece_score(conf, correct)
    ts = TemperatureScaler()
    ts.fit(logits, labels)
    logits_c = ts.transform(logits)
    conf_c = (logits_c.max(axis=1) / np.exp(logits_c).sum(axis=1)).astype(float)
    e1 = ece_score(conf_c, correct)
    assert e1 <= e0 + 1e-6
