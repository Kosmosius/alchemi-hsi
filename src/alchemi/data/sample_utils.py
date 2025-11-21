"""Helpers for moving between per-pixel :class:`Sample` objects and cubes."""

from __future__ import annotations

import numpy as np

from alchemi.data.cube import Cube
from alchemi.types import Sample

__all__ = ["cube_from_sample"]


def cube_from_sample(sample: Sample) -> Cube:
    """Create a 1x1 :class:`Cube` from a :class:`Sample` when compatible.

    This is primarily intended for converting lab-style spectra into the
    canonical cube representation so they can flow through ingestion pipelines
    that expect a spatial cube. The resulting cube preserves the sample's
    spectral axis, value kind, units, and sensor metadata.
    """

    values = np.asarray(sample.spectrum.values, dtype=np.float64).reshape(1, 1, -1)
    axis = np.asarray(sample.spectrum.wavelengths.nm, dtype=np.float64)

    sensor = None
    attrs_raw = sample.meta
    attrs = (
        dict(attrs_raw)
        if isinstance(attrs_raw, dict)
        else attrs_raw.as_dict()
    )
    if "sensor" in attrs:
        sensor = str(attrs["sensor"])

    cube = Cube(
        data=values,
        axis=axis,
        axis_unit="wavelength_nm",
        value_kind=sample.spectrum.kind.value,
        attrs=attrs,
    )
    if sensor is not None:
        cube.sensor = sensor
    cube.units = sample.spectrum.units
    return cube
