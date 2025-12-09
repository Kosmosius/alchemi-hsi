"""Uncertainty estimation utilities including calibration, ensembles, and OOD."""

from .calibration import (
    TemperatureLike,
    brier_score,
    calibration_metrics,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    temperature_scale_logits,
)
from .conformal import (
    classification_conformal_thresholds,
    classification_label_set,
    regression_conformal_thresholds,
    regression_interval,
)
from .ensembles import EnsembleOutput, TorchEnsemble, run_ensemble, train_ensemble
from .ood import (
    combine_ood_scores,
    energy_score,
    mahalanobis_distance,
    mahalanobis_ood_score,
    softmax_complement_score,
    spectral_angle_mapper,
)

__all__ = [
    "TemperatureLike",
    "brier_score",
    "calibration_metrics",
    "expected_calibration_error",
    "fit_temperature",
    "negative_log_likelihood",
    "temperature_scale_logits",
    "classification_conformal_thresholds",
    "classification_label_set",
    "regression_conformal_thresholds",
    "regression_interval",
    "EnsembleOutput",
    "TorchEnsemble",
    "run_ensemble",
    "train_ensemble",
    "combine_ood_scores",
    "energy_score",
    "mahalanobis_distance",
    "mahalanobis_ood_score",
    "softmax_complement_score",
    "spectral_angle_mapper",
]
