"""Adapter stub for AVIRIS-NG scenes.

This module mirrors the structure of :mod:`alchemi.data.adapters.emit` but
inserts TODO markers where mission documentation is required to parse official
products.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, List

import numpy as np
import xarray as xr

from alchemi.registry import srfs
from alchemi.spectral import Sample, Spectrum

from ..io import avirisng_pixel, load_avirisng_l1b

__all__ = ["iter_aviris_ng_pixels", "load_aviris_ng_scene"]


def _to_nanometers(wavelengths: np.ndarray) -> np.ndarray:
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    if wavelengths.max(initial=0.0) < 100.0:
        return wavelengths * 1_000.0
    return wavelengths


def iter_aviris_ng_pixels(path: str) -> Iterable[Sample]:
    """Iterate through an AVIRIS-NG cube yielding :class:`Sample` objects."""

    ds = load_avirisng_l1b(path)
    wavelengths = _to_nanometers(ds["wavelength"].values)
    srfs_for_sensor = None
    try:
        srfs_for_sensor = srfs.get_srf("aviris-ng")
    except Exception:
        srfs_for_sensor = None

    radiance = ds["radiance"]
    band_mask = np.asarray(ds["band_mask"].values, dtype=bool) if "band_mask" in ds else None

    # TODO: AVIRIS-NG L1B stores additional QA (clouds, smile). Wire these in.
    quality_base: Dict[str, np.ndarray] = {}
    if band_mask is not None:
        quality_base["band_mask"] = np.broadcast_to(band_mask, radiance.shape)

    for y in range(radiance.shape[0]):
        for x in range(radiance.shape[1]):
            values = np.asarray(radiance[y, x, :], dtype=np.float64)
            # Products are typically W/m^2/sr/µm; convert to per-nm.
            values = values / 1_000.0
            spec = Spectrum(wavelength_nm=wavelengths, values=values, kind="radiance")
            quality_masks = {name: mask[y, x, :] for name, mask in quality_base.items()}
            yield Sample(
                spectrum=spec,
                sensor_id="aviris-ng",
                quality_masks=quality_masks,
                srf_matrix=srfs_for_sensor,
                ancillary={"source_path": path, "y": int(y), "x": int(x)},
            )


def load_aviris_ng_scene(path: str) -> List[Sample]:
    """Materialise :func:`iter_aviris_ng_pixels` into a list."""

    return list(iter_aviris_ng_pixels(path))
