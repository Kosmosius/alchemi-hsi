"""Evaluation pipelines and metrics for the Alchemi project."""

from . import metrics
from .coverage_analysis import (
    geography_histogram,
    landcover_histogram,
    ontology_coverage,
    sensor_histogram,
    summarize_gaps,
)
from .gas_eval import enhancement_rmse, limit_of_detection_by_surface, plume_detection_metrics
from .heavy_atmosphere_eval import calibration_summary, split_by_regime, summarize_abstention
from .representation_eval import linear_probe_cross_sensor, recall_at_k
from .srf_robustness_eval import apply_srf_perturbations, robustness_degradation, sweep_perturbations
from .solids_eval import compare_dominant_minerals, compute_reconstruction_errors, limit_of_detection_experiment
from .teacher_noise_eval import calibration_by_teacher_confidence, compare_teacher_model_truth

__all__ = [
    "metrics",
    "compare_dominant_minerals",
    "compute_reconstruction_errors",
    "limit_of_detection_experiment",
    "plume_detection_metrics",
    "enhancement_rmse",
    "limit_of_detection_by_surface",
    "recall_at_k",
    "linear_probe_cross_sensor",
    "apply_srf_perturbations",
    "robustness_degradation",
    "sweep_perturbations",
    "split_by_regime",
    "summarize_abstention",
    "calibration_summary",
    "sensor_histogram",
    "geography_histogram",
    "landcover_histogram",
    "ontology_coverage",
    "summarize_gaps",
    "compare_teacher_model_truth",
    "calibration_by_teacher_confidence",
]
