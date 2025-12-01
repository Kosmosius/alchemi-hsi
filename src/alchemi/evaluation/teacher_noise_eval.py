"""Teacher noise evaluation utilities."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .metrics import expected_calibration_error, brier_score, prediction_set_coverage, rmse


def compare_teacher_model_truth(
    teacher_predictions: np.ndarray,
    model_predictions: np.ndarray,
    synthetic_truth: np.ndarray,
) -> Mapping[str, float]:
    """Compare teacher/model outputs against synthetic truth when available."""

    teacher_error = rmse(synthetic_truth, teacher_predictions)
    model_error = rmse(synthetic_truth, model_predictions)
    return {"teacher_rmse": teacher_error, "model_rmse": model_error}


def calibration_by_teacher_confidence(
    model_probabilities: np.ndarray,
    labels: Sequence[int],
    teacher_confidences: Sequence[float],
    bins: Sequence[float] | None = None,
) -> Mapping[str, float]:
    """Stratify coverage and calibration metrics by teacher confidence."""

    if bins is None:
        bins = [0.0, 0.25, 0.5, 0.75, 1.0]

    labels_arr = np.asarray(labels)
    confidences = np.asarray(teacher_confidences)
    bin_edges = np.asarray(bins)
    summary: dict[str, float] = {}
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue
        bucket_name = f"conf_{lower:.2f}_{upper:.2f}"
        bucket_probs = model_probabilities[mask]
        bucket_labels = labels_arr[mask]
        summary[f"ece_{bucket_name}"] = expected_calibration_error(bucket_probs, bucket_labels)
        summary[f"brier_{bucket_name}"] = brier_score(bucket_probs, bucket_labels)
        summary[f"coverage_{bucket_name}"] = prediction_set_coverage(
            [np.where(p > 1.0 / bucket_probs.shape[1])[0] for p in bucket_probs], bucket_labels
        )
    return summary


__all__ = ["compare_teacher_model_truth", "calibration_by_teacher_confidence"]
