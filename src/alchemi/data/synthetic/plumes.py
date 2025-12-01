"""Synthetic gas plume generators."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from alchemi.registry import srfs
from alchemi.spectral import Sample, Spectrum


def _generate_plume_mask(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    center = rng.uniform(low=[0.25 * h, 0.25 * w], high=[0.75 * h, 0.75 * w])
    sigma = rng.uniform(low=min(h, w) * 0.1, high=min(h, w) * 0.25)
    mask = np.exp(-(((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / (2 * sigma**2)))
    return mask / mask.max()


def generate_plume_samples(
    background: Sample,
    gas_signature: Spectrum,
    *,
    strength: float = 0.05,
    rng: np.random.Generator | None = None,
) -> List[Tuple[Sample, np.ndarray]]:
    """Inject a synthetic gas plume into a background sample."""

    rng = rng or np.random.default_rng()
    mask = _generate_plume_mask((8, 8), rng)
    plume_scale = strength * mask[..., np.newaxis]

    bg_values = background.spectrum.values[np.newaxis, np.newaxis, :]
    gas_interp = np.interp(background.spectrum.wavelength_nm, gas_signature.wavelength_nm, gas_signature.values)
    plume_cube = bg_values * (1 - plume_scale) + (bg_values + gas_interp) * plume_scale

    srfs_for_sensor = None
    try:
        srfs_for_sensor = srfs.get_srf(background.sensor_id)
    except Exception:
        # Fallback: leave SRF unset when registry lookup fails in stub environments.
        pass

    samples: list[Tuple[Sample, np.ndarray]] = []
    for y in range(plume_cube.shape[0]):
        for x in range(plume_cube.shape[1]):
            values = plume_cube[y, x, :]
            spectrum = Spectrum(
                wavelength_nm=background.spectrum.wavelength_nm,
                values=values,
                kind=background.spectrum.kind,
            )
            quality_masks = dict(background.quality_masks)
            plume_mask = np.isfinite(values)
            quality_masks["plume"] = plume_mask
            sample = Sample(
                spectrum=spectrum,
                sensor_id=background.sensor_id,
                quality_masks=quality_masks,
                srf_matrix=srfs_for_sensor,
                ancillary={"plume_strength": strength, "source_sample": background.ancillary},
            )
            samples.append((sample, mask[y, x]))

    return samples
