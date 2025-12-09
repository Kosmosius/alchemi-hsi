"""Common evaluation metrics used across Alchemi tasks.

The implementations are intentionally lightweight and avoid heavyweight ML
frameworks so they can be reused from both training-time hooks and simple
scripts. All functions accept numpy arrays and return python scalars or
small dictionaries.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise

import numpy as np

# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute simple classification accuracy."""

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(y_true_arr == y_pred_arr))


def _precision_recall_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, fn


def precision_recall_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    average: str = "macro",
) -> Mapping[str, float]:
    """Precision/recall/F1 for binary or multi-class targets.

    The ``average`` argument follows the scikit-learn semantics for ``macro``
    and ``micro`` averages. The function returns a small mapping to simplify
    structured logging.
    """

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels = np.unique(y_true_arr)

    if average not in {"macro", "micro"}:
        raise ValueError("average must be 'macro' or 'micro'")

    if average == "micro":
        tp = int(np.sum(y_true_arr == y_pred_arr))
        fp = int(np.sum(y_true_arr != y_pred_arr))
        fn = fp
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}

    # Macro average: compute per-class then average
    precisions = []
    recalls = []
    f1s = []
    for label in labels:
        tp, fp, fn = _precision_recall_counts(y_true_arr == label, y_pred_arr == label)
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
    }


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root-mean-squared error."""

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean absolute error."""

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Coefficient of determination."""

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------


def intersection_over_union(y_true_mask: np.ndarray, y_pred_mask: np.ndarray) -> float:
    """Compute IoU for boolean or integer masks."""

    y_true_bool = np.asarray(y_true_mask).astype(bool)
    y_pred_bool = np.asarray(y_pred_mask).astype(bool)
    intersection = np.logical_and(y_true_bool, y_pred_bool).sum()
    union = np.logical_or(y_true_bool, y_pred_bool).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def segmentation_precision_recall(
    y_true_mask: np.ndarray, y_pred_mask: np.ndarray
) -> Mapping[str, float]:
    """Precision/recall for pixel-level detections."""

    y_true_bool = np.asarray(y_true_mask).astype(bool)
    y_pred_bool = np.asarray(y_pred_mask).astype(bool)
    tp = np.logical_and(y_true_bool, y_pred_bool).sum()
    fp = np.logical_and(~y_true_bool, y_pred_bool).sum()
    fn = np.logical_and(y_true_bool, ~y_pred_bool).sum()
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return {"precision": float(precision), "recall": float(recall)}


# ---------------------------------------------------------------------------
# Spectral metrics
# ---------------------------------------------------------------------------


def spectral_angle_mapper(reference: np.ndarray, target: np.ndarray) -> float:
    """Compute SAM between two spectra in radians."""

    ref = np.asarray(reference, dtype=float)
    tgt = np.asarray(target, dtype=float)
    numerator = np.dot(ref, tgt)
    denom = np.linalg.norm(ref) * np.linalg.norm(tgt) + 1e-12
    cosine = np.clip(numerator / denom, -1.0, 1.0)
    return float(np.arccos(cosine))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between vectors."""

    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-12))


def residual_norm(reference: np.ndarray, target: np.ndarray, order: int = 2) -> float:
    """Residual norm between spectra."""

    ref = np.asarray(reference, dtype=float)
    tgt = np.asarray(target, dtype=float)
    return float(np.linalg.norm(ref - tgt, ord=order))


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def expected_calibration_error(
    probabilities: np.ndarray, labels: Sequence[int], n_bins: int = 10
) -> float:
    """Expected calibration error for probabilistic classification outputs."""

    probs = np.asarray(probabilities, dtype=float)
    labels_arr = np.asarray(labels)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for bin_lower, bin_upper in pairwise(bins):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if not np.any(in_bin):
            continue
        acc = np.mean(predictions[in_bin] == labels_arr[in_bin])
        conf = np.mean(confidences[in_bin])
        ece += np.abs(acc - conf) * np.mean(in_bin)
    return float(ece)


def negative_log_likelihood(probabilities: np.ndarray, labels: Sequence[int]) -> float:
    """Average negative log-likelihood for classification."""

    probs = np.asarray(probabilities, dtype=float)
    labels_arr = np.asarray(labels)
    eps = 1e-12
    clipped = np.clip(probs, eps, 1.0)
    chosen = clipped[np.arange(len(labels_arr)), labels_arr]
    return float(-np.mean(np.log(chosen)))


def brier_score(probabilities: np.ndarray, labels: Sequence[int]) -> float:
    """Multi-class Brier score."""

    probs = np.asarray(probabilities, dtype=float)
    labels_arr = np.asarray(labels)
    num_classes = probs.shape[1]
    one_hot = np.eye(num_classes)[labels_arr]
    return float(np.mean((probs - one_hot) ** 2))


def prediction_set_coverage(
    prediction_sets: Sequence[Iterable[int]], labels: Sequence[int]
) -> float:
    """Coverage rate for prediction sets or abstention-enabled classifiers."""

    label_arr = np.asarray(labels)
    covered = [
        label in pred_set
        for label, pred_set in zip(label_arr, prediction_sets, strict=False)
    ]
    return float(np.mean(covered))


__all__ = [
    "accuracy",
    "brier_score",
    "cosine_similarity",
    "expected_calibration_error",
    "intersection_over_union",
    "mae",
    "negative_log_likelihood",
    "precision_recall_f1",
    "prediction_set_coverage",
    "r2_score",
    "residual_norm",
    "rmse",
    "segmentation_precision_recall",
    "spectral_angle_mapper",
]
