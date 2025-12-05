"""Adapter utilities for AVIRIS-NG samples."""

from __future__ import annotations

import numpy as np
import xarray as xr

from alchemi.data.io import avirisng_pixel
from alchemi.spectral import Sample

from .aviris_ng import _build_sample

__all__ = ["load_avirisng_pixel"]


def load_avirisng_pixel(ds: xr.Dataset, position: tuple[int, int]) -> Sample:
    """Create a canonical :class:`~alchemi.spectral.Sample` from a dataset and (y, x) index."""

    y, x = position
    spectrum = avirisng_pixel(ds, y=y, x=x)
    values = np.asarray(spectrum.values, dtype=np.float64)
    path = str(getattr(ds, "encoding", {}).get("source", ""))
    return _build_sample(ds=ds, path=path, y=y, x=x, values=values, spectrum_kind=str(spectrum.kind))
