"""Helpers for moving between per-pixel :class:`Sample` objects and cubes."""

from __future__ import annotations

import numpy as np

from alchemi.data.cube import Cube
from alchemi.spectral import Sample

__all__ = ["cube_from_sample"]


def cube_from_sample(sample: Sample) -> Cube:
    """Create a 1x1 :class:`Cube` from a :class:`Sample` when compatible.

    This is primarily intended for converting lab-style spectra into the
    canonical cube representation so they can flow through ingestion pipelines
    that expect a spatial cube. The resulting cube preserves the sample's
    spectral axis, value kind, units, and sensor metadata.
    """

    values = np.asarray(sample.spectrum.values, dtype=np.float64).reshape(1, 1, -1)
    axis = np.asarray(sample.spectrum.wavelength_nm, dtype=np.float64)

    attrs = {"sensor": sample.sensor_id, **sample.ancillary}

    cube = Cube(
        data=values,
        axis=axis,
        axis_unit="wavelength_nm",
        value_kind=sample.spectrum.kind,
        attrs=attrs,
    )
    cube.sensor = sample.sensor_id
    cube.units = None
    return cube
