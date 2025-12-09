"""Evaluation utilities for trace gas detection tasks."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from .metrics import intersection_over_union, segmentation_precision_recall, rmse


def plume_detection_metrics(
    predicted_mask: np.ndarray, reference_mask: np.ndarray
) -> Mapping[str, float]:
    """Compute IoU and precision/recall for plume detection masks."""

    iou = intersection_over_union(reference_mask, predicted_mask)
    pr = segmentation_precision_recall(reference_mask, predicted_mask)
    return {"iou": iou, **pr}


def enhancement_rmse(predicted_enhancement: np.ndarray, teacher_enhancement: np.ndarray) -> float:
    """RMSE between model and teacher enhancement fields."""

    return rmse(teacher_enhancement, predicted_enhancement)


def limit_of_detection_by_surface(
    plume_strengths: Sequence[float],
    detections: Sequence[bool],
    surface_types: Sequence[str],
    snr_values: Sequence[float],
    snr_bins: Sequence[float] | None = None,
) -> Mapping[str, Mapping[str, float]]:
    """Estimate detection probability as a function of surface type and SNR."""

    strengths = np.asarray(plume_strengths)
    detects = np.asarray(detections).astype(bool)
    surfaces = np.asarray(surface_types)
    snr_arr = np.asarray(snr_values)

    if snr_bins is None:
        snr_bins = [0, 50, 100, 150, 200, np.inf]

    summary: dict[str, Mapping[str, float]] = defaultdict(dict)
    for surface in np.unique(surfaces):
        surface_mask = surfaces == surface
        for lower, upper in zip(snr_bins[:-1], snr_bins[1:]):
            in_bin = surface_mask & (snr_arr >= lower) & (snr_arr < upper)
            if not np.any(in_bin):
                continue
            detection_rate = float(np.mean(detects[in_bin]))
            avg_strength = float(np.mean(strengths[in_bin]))
            label = f"snr_{lower:.0f}_{upper:.0f}"
            summary[surface][label] = detection_rate
            summary[surface][f"avg_strength_{label}"] = avg_strength
    return summary


__all__ = [
    "plume_detection_metrics",
    "enhancement_rmse",
    "limit_of_detection_by_surface",
]
