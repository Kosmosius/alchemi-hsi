"""Adapter stub for HyTES brightness-temperature chips."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, List

import numpy as np
import xarray as xr

from alchemi.registry import srfs
from alchemi.spectral import Sample, Spectrum

from ..io import load_hytes_l1b

__all__ = ["iter_hytes_pixels", "load_hytes_scene"]


def _load_ds(path: str) -> xr.Dataset:
    try:
        return load_hytes_l1b(path)
    except Exception:  # pragma: no cover - placeholder for optional dependency errors
        # TODO: Implement HDF5/GeoTIFF parsing once HyTES documentation is wired in.
        return xr.open_dataset(path)


def iter_hytes_pixels(path: str) -> Iterable[Sample]:
    """Yield HyTES pixels as :class:`Sample` objects.

    HyTES products are typically stored as brightness temperature in Kelvin. No
    unit conversion is applied here besides a defensive cast to ``float64``.
    """

    ds = _load_ds(path)
    bt = ds["brightness_temperature"] if "brightness_temperature" in ds else ds["bt"]
    wavelengths = np.asarray(ds["wavelength_nm"].values, dtype=np.float64)

    srfs_for_sensor = None
    try:
        srfs_for_sensor = srfs.get_srf("hytes")
    except Exception:
        srfs_for_sensor = None
    band_mask = np.asarray(ds["band_mask"].values, dtype=bool) if "band_mask" in ds else None

    quality_base: Dict[str, np.ndarray] = {}
    if band_mask is not None:
        quality_base["band_mask"] = np.broadcast_to(band_mask, bt.shape)

    for y in range(bt.shape[0]):
        for x in range(bt.shape[1]):
            values = np.asarray(bt[y, x, :], dtype=np.float64)
            spectrum = Spectrum(wavelength_nm=wavelengths, values=values, kind="BT")
            quality_masks = {name: mask[y, x, :] for name, mask in quality_base.items()}
            yield Sample(
                spectrum=spectrum,
                sensor_id="hytes",
                quality_masks=quality_masks,
                srf_matrix=srfs_for_sensor,
                ancillary={"source_path": path, "y": int(y), "x": int(x)},
            )


def load_hytes_scene(path: str) -> List[Sample]:
    """Materialise an iterator of HyTES pixels into memory."""

    return list(iter_hytes_pixels(path))
