"""Synthetic solid mixture generation for LoD/unmixing tasks."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from alchemi.physics.resampling import convolve_to_bands
from alchemi.registry import ontology, srfs
from alchemi.spectral import Sample, Spectrum


def _sample_library_entries(categories: Iterable[str], rng: np.random.Generator) -> list[Spectrum]:
    spectra: list[Spectrum] = []
    for category in categories:
        try:
            entry = ontology.pick_random_spectrum(category, rng=rng)
        except Exception:
            # TODO: replace with proper ontology sampling once the registry is wired up.
            wavelengths = np.linspace(350, 2500, 200)
            values = rng.random(wavelengths.size)
            entry = Spectrum(wavelength_nm=wavelengths, values=values, kind="reflectance")
        if hasattr(entry, "to_spectrum"):
            entry = entry.to_spectrum()
        spectra.append(entry)
    return spectra


def _sample_abundances(num_endmembers: int, rng: np.random.Generator) -> np.ndarray:
    abundances = rng.random(num_endmembers + 1)  # final slot for shade/residual
    abundances /= abundances.sum()
    return abundances


def generate_mixture_samples(
    categories: Iterable[str],
    sensor_id: str,
    *,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> List[Tuple[Sample, np.ndarray]]:
    """Generate synthetic mixtures convolved to a sensor SRF."""

    rng = rng or np.random.default_rng()
    endmembers = _sample_library_entries(categories, rng)
    abundances = _sample_abundances(len(endmembers), rng)

    wavelength_grid = endmembers[0].wavelength_nm
    stacked = np.stack([em.values for em in endmembers], axis=0)
    mixed = abundances[:-1] @ stacked
    mixed += abundances[-1] * 0.0  # residual channel placeholder

    if noise_sigma > 0:
        mixed = mixed + rng.normal(scale=noise_sigma, size=mixed.shape)

    srf = srfs.get_srf(sensor_id)
    band_values = _convolve_to_bands(srf, wavelength_grid, mixed)
    spectrum = Spectrum(wavelength_nm=srf.centers_nm, values=band_values, kind="reflectance")
    sample = Sample(spectrum=spectrum, sensor_id=sensor_id, ancillary={"abundances": abundances})
    return [(sample, abundances)]


def _convolve_to_bands(srf: object, wavelengths: np.ndarray, values: np.ndarray) -> np.ndarray:
    spectrum = Spectrum(wavelength_nm=wavelengths, values=values, kind="reflectance")
    convolved = convolve_to_bands(spectrum, srf)
    return np.asarray(convolved.values, dtype=np.float64)
