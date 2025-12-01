"""Spectral response function robustness experiments."""

from __future__ import annotations

from typing import Callable, Iterable, Mapping

import numpy as np

from .metrics import spectral_angle_mapper


def apply_srf_perturbations(
    spectra: np.ndarray,
    center_shift: float = 0.0,
    fwhm_scale: float = 1.0,
    distortion: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Apply simple SRF perturbations via wavelength shifting and smoothing."""

    num_samples, num_bands = spectra.shape
    wavelengths = np.linspace(0, 1, num_bands)
    shifted = wavelengths + center_shift
    shifted = np.clip(shifted, 0, 1)
    # Simple resampling
    perturbed = np.vstack([np.interp(wavelengths, shifted, spec) for spec in spectra])

    if fwhm_scale != 1.0:
        kernel_width = max(int(3 * fwhm_scale), 1)
        kernel = np.exp(-0.5 * (np.arange(-kernel_width, kernel_width + 1) / fwhm_scale) ** 2)
        kernel = kernel / kernel.sum()
        padded = np.pad(perturbed, ((0, 0), (kernel_width, kernel_width)), mode="edge")
        smoothed = np.vstack([
            np.convolve(row, kernel, mode="valid") for row in padded
        ])
        perturbed = smoothed

    if distortion is not None:
        perturbed = distortion(perturbed)

    return perturbed


def robustness_degradation(
    baseline_spectra: np.ndarray,
    perturbed_spectra: np.ndarray,
    evaluator: Callable[[np.ndarray, np.ndarray], float] = spectral_angle_mapper,
) -> float:
    """Measure degradation between baseline and perturbed spectra using a metric."""

    scores = [evaluator(base, pert) for base, pert in zip(baseline_spectra, perturbed_spectra)]
    return float(np.mean(scores))


def sweep_perturbations(
    spectra: np.ndarray,
    perturbation_settings: Iterable[Mapping[str, float]],
    evaluator: Callable[[np.ndarray, np.ndarray], float] = spectral_angle_mapper,
) -> list[Mapping[str, float]]:
    """Apply a list of perturbations and summarize metric degradation."""

    baseline = spectra.copy()
    results: list[Mapping[str, float]] = []
    for settings in perturbation_settings:
        perturbed = apply_srf_perturbations(baseline, **settings)
        degradation = robustness_degradation(baseline, perturbed, evaluator=evaluator)
        results.append({**settings, "degradation": degradation})
    return results


__all__ = ["apply_srf_perturbations", "robustness_degradation", "sweep_perturbations"]
