"""Adapter utilities for AVIRIS-NG samples."""

from __future__ import annotations

import numpy as np
import xarray as xr

from alchemi.data.io import avirisng_pixel
from alchemi.spectral import Sample

__all__ = ["load_avirisng_pixel"]


def load_avirisng_pixel(ds: xr.Dataset, position: tuple[int, int]) -> Sample:
    """Create a canonical :class:`~alchemi.spectral.Sample` from a dataset and (y, x) index."""

    y, x = position
    spectrum = avirisng_pixel(ds, y=y, x=x)
    wavelengths = np.asarray(ds["wavelength_nm"].values, dtype=float)
    fwhm = np.asarray(ds["fwhm_nm"].values, dtype=float) if "fwhm_nm" in ds else np.full_like(wavelengths, np.nan)
    band_mask = np.asarray(ds["band_mask"].values, dtype=bool) if "band_mask" in ds else np.ones_like(wavelengths, dtype=bool)

    return Sample(
        spectrum=spectrum,
        sensor_id=ds.attrs.get("sensor", "aviris-ng"),
        acquisition_time=ds.attrs.get("datetime"),
        band_meta={
            "center_nm": wavelengths,
            "width_nm": fwhm,
            "valid_mask": band_mask,
            "srf_source": np.array(["mission"] * wavelengths.size),
        },
        quality_masks={"band_mask": band_mask},
        ancillary={"source_path": str(getattr(ds, "encoding", {}).get("source", "")), "y": int(y), "x": int(x)},
    )
