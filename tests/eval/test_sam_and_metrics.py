import numpy as np
import pytest

from alchemi.training.calibration import TemperatureScaler
from alchemi.training.metrics import ece_score, spectral_angle


def _softmax_confidences(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    probs = exps / np.maximum(exps.sum(axis=1, keepdims=True), 1e-12)
    return probs.max(axis=1).astype(float)


def test_spectral_angle_invariant_to_scaling():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    assert spectral_angle(x, y) == pytest.approx(0.0, abs=1e-8)


def test_spectral_angle_known_angles():
    assert spectral_angle(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])) == pytest.approx(
        np.pi / 2, abs=1e-8
    )

    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])  # cos(theta) = 0.5
    assert spectral_angle(v1, v2) == pytest.approx(np.arccos(0.5), abs=1e-8)


def test_spectral_angle_ignores_nan_bands():
    x = np.array([1.0, np.nan, 0.0])
    y = np.array([2.0, np.nan, 0.0])
    assert spectral_angle(x, y) == pytest.approx(0.0, abs=1e-8)

    x = np.array([1.0, np.nan, 0.0])
    y = np.array([0.0, np.nan, 1.0])
    assert spectral_angle(x, y) == pytest.approx(np.pi / 2, abs=1e-8)


def test_temperature_scaling_reduces_ece_and_preserves_ranking():
    logits_cal = np.array(
        [[6.0, 0.0], [5.0, 0.0], [0.0, 4.0], [0.0, 5.0], [3.0, 0.0], [0.0, 3.0]]
    )
    labels_cal = np.array([0, 0, 1, 1, 0, 1])

    logits_eval = np.array(
        [[4.0, 0.0], [2.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 1.0], [3.0, 0.0]]
    )
    labels_eval = np.array([0, 0, 1, 0, 1, 0])

    scaler = TemperatureScaler()
    scaler.fit(logits_cal, labels_cal)

    conf_before = _softmax_confidences(logits_eval)
    logits_scaled = scaler.transform(logits_eval)
    conf_after = _softmax_confidences(logits_scaled)

    correct = (logits_eval.argmax(axis=1) == labels_eval).astype(float)

    ece_before = ece_score(conf_before, correct)
    ece_after = ece_score(conf_after, correct)

    assert ece_after < ece_before - 0.01
    assert np.array_equal(logits_eval.argmax(axis=1), logits_scaled.argmax(axis=1))

