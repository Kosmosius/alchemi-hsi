"""Spectral response function robustness experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Literal

import numpy as np

from alchemi.registry import srfs
from alchemi.srf.registry import sensor_srf_from_legacy
from alchemi.srf.resample import resample_values_with_srf
from alchemi.srf.synthetic import SRFJitterConfig, jitter_sensor_srf
from alchemi.srf.utils import load_sensor_srf
from alchemi.types import SRFMatrix

from .metrics import spectral_angle_mapper


def apply_srf_perturbations(
    spectra: np.ndarray,
    *,
    perturbation_mode: Literal["axis", "srf"] = "axis",
    center_shift: float = 0.0,
    fwhm_scale: float = 1.0,
    distortion: Callable[[np.ndarray], np.ndarray] | None = None,
    # SRF-matrix mode inputs
    wavelength_grid_nm: np.ndarray | None = None,
    sensor_id: str | None = None,
    base_sensor_srf: SRFMatrix | None = None,
    center_shift_std_nm: float = 0.0,
    width_scale_std: float = 0.0,
    shape_jitter_std: float = 0.0,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    return_srf: bool = False,
) -> np.ndarray | tuple[np.ndarray, SRFMatrix | None]:
    """Apply SRF perturbations to spectra.

    Parameters
    ----------
    spectra:
        Spectral values shaped ``[N, L]`` on a high-resolution wavelength grid.
    perturbation_mode:
        "axis" retains the legacy behaviour (approximate shifts on a normalized
        grid). "srf" perturbs actual SRF matrices and re-convolves ``spectra``.

    Returns
    -------
    perturbed_spectra, perturbed_srf
        The perturbed observations and the SRF used to generate them
        (``perturbed_srf`` is ``None`` in "axis" mode).
    """

    if perturbation_mode == "axis":
        _num_samples, num_bands = spectra.shape
        wavelengths = np.linspace(0, 1, num_bands)
        shifted = wavelengths + center_shift
        shifted = np.clip(shifted, 0, 1)
        perturbed = np.vstack([np.interp(wavelengths, shifted, spec) for spec in spectra])

        if fwhm_scale != 1.0:
            kernel_width = max(int(3 * fwhm_scale), 1)
            kernel = np.exp(-0.5 * (np.arange(-kernel_width, kernel_width + 1) / fwhm_scale) ** 2)
            kernel = kernel / kernel.sum()
            padded = np.pad(perturbed, ((0, 0), (kernel_width, kernel_width)), mode="edge")
            smoothed = np.vstack([np.convolve(row, kernel, mode="valid") for row in padded])
            perturbed = smoothed

        if distortion is not None:
            perturbed = distortion(perturbed)

        return (perturbed, None) if return_srf else perturbed

    if perturbation_mode != "srf":  # pragma: no cover - defensive guard
        raise ValueError(f"Unsupported perturbation_mode={perturbation_mode!r}")

    if wavelength_grid_nm is None:
        raise ValueError("wavelength_grid_nm must be provided for 'srf' mode")

    resolved_srf = base_sensor_srf
    if resolved_srf is None:
        resolved_srf = load_sensor_srf(sensor_id) or srfs.get_sensor_srf(sensor_id or "")
    if resolved_srf is None:
        raise ValueError("A sensor SRF must be provided for 'srf' mode")

    if not hasattr(resolved_srf, "band_centers_nm"):
        resolved_srf = sensor_srf_from_legacy(resolved_srf)

    jitter_cfg = SRFJitterConfig(
        enabled=True,
        center_shift_std_nm=center_shift_std_nm,
        width_scale_std=width_scale_std,
        shape_jitter_std=shape_jitter_std,
        seed=seed,
        rng=rng,
    )
    perturbed_srf = jitter_sensor_srf(resolved_srf, jitter_cfg, rng=rng)

    perturbed, _ = resample_values_with_srf(spectra, wavelength_grid_nm, perturbed_srf)

    return (perturbed, perturbed_srf) if return_srf else perturbed


def robustness_degradation(
    baseline_spectra: np.ndarray,
    perturbed_spectra: np.ndarray,
    evaluator: Callable[[np.ndarray, np.ndarray], float] = spectral_angle_mapper,
) -> float:
    """Measure degradation between baseline and perturbed spectra using a metric."""

    scores = [
        evaluator(base, pert)
        for base, pert in zip(baseline_spectra, perturbed_spectra, strict=False)
    ]
    return float(np.mean(scores))


def sweep_perturbations(
    spectra: np.ndarray,
    perturbation_settings: Iterable[Mapping[str, float]],
    evaluator: Callable[[np.ndarray, np.ndarray], float] = spectral_angle_mapper,
    *,
    perturbation_mode: Literal["axis", "srf"] = "axis",
    wavelength_grid_nm: np.ndarray | None = None,
    sensor_id: str | None = None,
    base_sensor_srf: SRFMatrix | None = None,
) -> list[Mapping[str, float]]:
    """Apply a list of perturbations and summarize metric degradation."""

    if perturbation_mode == "srf":
        if wavelength_grid_nm is None:
            raise ValueError("wavelength_grid_nm must be provided for 'srf' mode")

        resolved_srf = base_sensor_srf
        if resolved_srf is None:
        resolved_srf = load_sensor_srf(sensor_id) or srfs.get_sensor_srf(sensor_id or "")
        if resolved_srf is None:
            raise ValueError("A sensor SRF must be provided for 'srf' mode")

        if not hasattr(resolved_srf, "band_centers_nm"):
            resolved_srf = sensor_srf_from_legacy(resolved_srf)

        baseline, _ = resample_values_with_srf(spectra, wavelength_grid_nm, resolved_srf)
        base_sensor_srf = resolved_srf
    else:
        baseline = spectra.copy()

    results: list[Mapping[str, float]] = []
    for settings in perturbation_settings:
        result = apply_srf_perturbations(
            spectra,
            perturbation_mode=perturbation_mode,
            wavelength_grid_nm=wavelength_grid_nm,
            sensor_id=sensor_id,
            base_sensor_srf=base_sensor_srf,
            return_srf=True,
            **settings,
        )
        perturbed, perturbed_srf = result
        degradation = robustness_degradation(baseline, perturbed, evaluator=evaluator)
        enriched = {**settings, "degradation": degradation}
        if perturbed_srf is not None:
            enriched["perturbed_centers_nm"] = np.asarray(
                getattr(perturbed_srf, "band_centers_nm", None), dtype=np.float64
            )
            enriched["perturbed_widths_nm"] = np.asarray(
                getattr(perturbed_srf, "band_widths_nm", None), dtype=np.float64
            )
        results.append(enriched)
    return results


__all__ = ["apply_srf_perturbations", "robustness_degradation", "sweep_perturbations"]
