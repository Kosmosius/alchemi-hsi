"""Evaluation splits focused on heavy-atmosphere regimes."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from .metrics import expected_calibration_error, prediction_set_coverage, brier_score


def split_by_regime(
    regimes: Sequence[str],
    predictions: np.ndarray,
    labels: Sequence[int],
    metric_fn: Callable[[np.ndarray, Sequence[int]], float] = brier_score,
) -> Mapping[str, float]:
    """Compute metrics separately for trusted vs heavy-atmosphere regimes."""

    regimes_arr = np.asarray(regimes)
    labels_arr = np.asarray(labels)
    metrics: dict[str, float] = {}
    for regime_name in np.unique(regimes_arr):
        mask = regimes_arr == regime_name
        metrics[regime_name] = metric_fn(predictions[mask], labels_arr[mask])
    return metrics


def summarize_abstention(
    prediction_sets: Sequence[Sequence[int]],
    regimes: Sequence[str],
    labels: Sequence[int],
) -> Mapping[str, float]:
    """Summarize abstention rates and coverage per radiative-transfer regime."""

    regimes_arr = np.asarray(regimes)
    coverage = prediction_set_coverage(prediction_sets, labels)
    summary = {"overall_coverage": coverage}
    for regime_name in np.unique(regimes_arr):
        mask = regimes_arr == regime_name
        regime_sets = [p for p, m in zip(prediction_sets, mask) if m]
        regime_labels = [l for l, m in zip(labels, mask) if m]
        summary[f"coverage_{regime_name}"] = prediction_set_coverage(regime_sets, regime_labels)
        summary[f"abstention_rate_{regime_name}"] = float(1.0 - summary[f"coverage_{regime_name}"])
    return summary


def calibration_summary(
    predictions: np.ndarray,
    labels: Sequence[int],
    regimes: Sequence[str],
    n_bins: int = 10,
) -> Mapping[str, float]:
    """Compare calibration metrics across regimes."""

    regimes_arr = np.asarray(regimes)
    labels_arr = np.asarray(labels)
    summary: dict[str, float] = {
        "overall_ece": expected_calibration_error(predictions, labels_arr, n_bins=n_bins),
        "overall_brier": brier_score(predictions, labels_arr),
    }
    for regime_name in np.unique(regimes_arr):
        mask = regimes_arr == regime_name
        summary[f"ece_{regime_name}"] = expected_calibration_error(
            predictions[mask], labels_arr[mask], n_bins=n_bins
        )
        summary[f"brier_{regime_name}"] = brier_score(predictions[mask], labels_arr[mask])
    return summary


__all__ = ["split_by_regime", "summarize_abstention", "calibration_summary"]
