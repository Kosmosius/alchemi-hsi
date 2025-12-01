import numpy as np

from alchemi.evaluation.metrics import accuracy, mae, precision_recall_f1, spectral_angle_mapper


def test_classification_metrics_for_synthetic_predictions():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    acc = accuracy(y_true, y_pred)
    prf = precision_recall_f1(y_true, y_pred)
    assert acc == 0.75
    assert 0.0 <= prf["precision"] <= 1.0
    assert 0.0 <= prf["recall"] <= 1.0


def test_spectral_and_regression_metrics():
    ref = np.array([1.0, 1.0, 1.0])
    target = np.array([1.0, 0.5, 0.0])
    angle = spectral_angle_mapper(ref, target)
    assert angle > 0
    assert angle < np.pi / 2

    values_true = [0.0, 1.0, 2.0]
    values_pred = [0.1, 0.9, 1.8]
    error = mae(values_true, values_pred)
    assert 0 <= error < 0.2
