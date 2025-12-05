"""Helpers for moving between per-pixel :class:`Sample` objects and cubes."""

from __future__ import annotations

import numpy as np

from alchemi.data.cube import Cube
from alchemi.spectral import Sample

__all__ = ["compute_usable_mask", "cube_from_sample"]


def compute_usable_mask(sample: Sample, *, for_task: str = "default") -> np.ndarray:
    """Return a boolean mask of bands that are usable for downstream tasks.

    Quality masks follow a small convention across adapters:

    - ``valid_band`` (required): ``True`` where a band is generally usable.
    - ``deep_water_vapour`` (optional): ``True`` for deep absorption windows that
      many tasks exclude by default.
    - ``bad_detector`` (optional): ``True`` where detector issues are present.
    - Mission-specific masks such as ``cloud`` or ``saturation`` may also be
      present and can be incorporated into task-specific policies in the future.

    Parameters
    ----------
    sample:
        Sample containing ``quality_masks`` to combine.
    for_task:
        Optional hint for masking policy. The default excludes deep water-vapour
        windows when present; ``"all"`` keeps all valid bands.
    """

    quality = sample.quality_masks
    if "valid_band" not in quality:
        msg = "Sample is missing required 'valid_band' quality mask"
        raise KeyError(msg)

    usable = np.asarray(quality["valid_band"], dtype=bool).copy()

    exclusions: dict[str, tuple[str, ...]] = {
        "default": ("deep_water_vapour",),
        "solids": ("deep_water_vapour",),
        "gas": ("deep_water_vapour",),
        "all": (),
    }
    masks_to_drop = exclusions.get(for_task, exclusions["default"])

    for name in masks_to_drop:
        mask = quality.get(name)
        if mask is None:
            continue
        usable &= ~np.asarray(mask, dtype=bool)

    return usable


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
