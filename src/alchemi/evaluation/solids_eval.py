"""Evaluation helpers for solid mineral retrieval tasks."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from .metrics import precision_recall_f1, spectral_angle_mapper, residual_norm


def compare_dominant_minerals(
    predicted_labels: Sequence[int],
    predicted_abundances: Sequence[float],
    reference_labels: Sequence[int],
    reference_abundances: Sequence[float],
) -> Mapping[str, float]:
    """Compare dominant mineral predictions against EMIT L2B-style references."""

    label_accuracy = precision_recall_f1(reference_labels, predicted_labels, average="macro")
    abundance_error = float(
        np.mean(np.abs(np.asarray(predicted_abundances) - np.asarray(reference_abundances)))
    )
    return {**label_accuracy, "abundance_mae": abundance_error}


def compute_reconstruction_errors(
    predicted_spectra: np.ndarray,
    reference_spectra: np.ndarray,
    diagnostic_bands: Iterable[slice] | None = None,
) -> Mapping[str, float]:
    """Return SAM and band-depth reconstruction errors."""

    sam_scores = [
        spectral_angle_mapper(ref, pred) for ref, pred in zip(reference_spectra, predicted_spectra)
    ]
    residuals = [
        residual_norm(ref, pred) for ref, pred in zip(reference_spectra, predicted_spectra)
    ]

    band_depth_errors: list[float] = []
    if diagnostic_bands is not None:
        for band_slice in diagnostic_bands:
            ref_depth = np.abs(reference_spectra[:, band_slice].min(axis=1))
            pred_depth = np.abs(predicted_spectra[:, band_slice].min(axis=1))
            band_depth_errors.append(float(np.mean(np.abs(ref_depth - pred_depth))))

    metrics: dict[str, float] = {
        "sam": float(np.mean(sam_scores)),
        "residual_norm": float(np.mean(residuals)),
    }
    if band_depth_errors:
        metrics["band_depth_mae"] = float(np.mean(band_depth_errors))
    return metrics


def limit_of_detection_experiment(
    abundances: Sequence[float],
    detections: Sequence[bool],
    abundance_bins: Sequence[float] | None = None,
) -> Mapping[str, float]:
    """Summarize detection probability vs. abundance curves."""

    abundances_arr = np.asarray(abundances)
    detections_arr = np.asarray(detections).astype(bool)

    if abundance_bins is None:
        # default bins at 0%, 1%, 2.5%, 5%, 10%, 20%
        abundance_bins = [0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 1.0]

    bin_edges = np.asarray(abundance_bins)
    detection_rates = []
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (abundances_arr >= lower) & (abundances_arr < upper)
        if not np.any(in_bin):
            continue
        detection_rates.append(float(np.mean(detections_arr[in_bin])))

    lod_threshold = None
    for bound, rate in zip(bin_edges[1:], detection_rates):
        if rate >= 0.5:
            lod_threshold = bound
            break

    summary = {
        "detection_rates_mean": float(np.mean(detection_rates)) if detection_rates else 0.0,
        "lod50_abundance": float(lod_threshold) if lod_threshold is not None else float("nan"),
    }
    return summary


__all__ = [
    "compare_dominant_minerals",
    "compute_reconstruction_errors",
    "limit_of_detection_experiment",
]
