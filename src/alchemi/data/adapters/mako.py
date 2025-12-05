"""Adapter utilities for COMEX Mako samples."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from alchemi.io import mako_pixel_bt, mako_pixel_radiance
from alchemi.spectral import Sample

__all__ = ["load_mako_pixel", "load_mako_pixel_bt"]


def load_mako_pixel(ds: xr.Dataset, position: tuple[int, int]) -> Sample:
    """Create a canonical :class:`~alchemi.spectral.Sample` from a Mako L2S dataset."""

    spectrum = mako_pixel_radiance(ds, position[0], position[1])
    return _to_sample(ds, spectrum, position)


def load_mako_pixel_bt(ds: xr.Dataset, position: tuple[int, int]) -> Sample:
    """Create a canonical :class:`~alchemi.spectral.Sample` from a Mako BTEMP dataset."""

    spectrum = mako_pixel_bt(ds, position[0], position[1])
    return _to_sample(ds, spectrum, position)


def _to_sample(ds: xr.Dataset, spectrum: Any, position: tuple[int, int]) -> Sample:
    wavelengths = np.asarray(ds["wavelength_nm"].values, dtype=float) if "wavelength_nm" in ds else spectrum.wavelength_nm
    band_mask = np.asarray(ds["band_mask"].values, dtype=bool) if "band_mask" in ds else np.ones_like(wavelengths, dtype=bool)
    return Sample(
        spectrum=spectrum,
        sensor_id="mako",
        acquisition_time=ds.attrs.get("datetime"),
        band_meta={
            "center_nm": wavelengths,
            "width_nm": np.full_like(wavelengths, np.nan, dtype=float),
            "valid_mask": band_mask,
            "srf_source": np.array(["mission"] * wavelengths.size),
        },
        quality_masks={"band_mask": band_mask},
        ancillary={"y": int(position[0]), "x": int(position[1])},
    )
