"""Adapter stub for EnMAP products."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, List

import numpy as np
import xarray as xr

from alchemi.registry import srfs
from alchemi.spectral import Sample, Spectrum

from ..io import enmap_pixel, load_enmap_l1b

__all__ = ["iter_enmap_pixels", "load_enmap_scene"]


def _load_scene(path_or_pair: str | tuple[str, str]) -> xr.Dataset:
    if isinstance(path_or_pair, tuple):
        # TODO: The VNIR/SWIR pairing should follow the official delivery format.
        return load_enmap_l1b(path_or_pair[0], path_or_pair[1])
    return xr.load_dataset(path_or_pair)


def iter_enmap_pixels(path_or_pair: str | tuple[str, str]) -> Iterable[Sample]:
    """Yield EnMAP pixels as :class:`Sample` instances."""

    ds = _load_scene(path_or_pair)
    wavelengths = np.asarray(ds["wavelength_nm"].values, dtype=np.float64)
    radiance = ds["radiance"]
    srfs_for_sensor = None
    try:
        srfs_for_sensor = srfs.get_srf("enmap")
    except Exception:
        srfs_for_sensor = None

    band_mask = np.asarray(ds["band_mask"].values, dtype=bool) if "band_mask" in ds else None
    fwhm = np.asarray(ds["fwhm_nm"].values, dtype=np.float64) if "fwhm_nm" in ds else None

    quality_base: Dict[str, np.ndarray] = {}
    if band_mask is not None:
        quality_base["band_mask"] = np.broadcast_to(band_mask, radiance.shape)

    # TODO: Fold in L1B QA flags once available in fixtures.
    for y in range(radiance.shape[0]):
        for x in range(radiance.shape[1]):
            values = np.asarray(radiance[y, x, :], dtype=np.float64)
            spec = Spectrum(wavelength_nm=wavelengths, values=values, kind="radiance")
            band_meta = None
            if fwhm is not None:
                band_meta = {
                    "center_nm": wavelengths,
                    "width_nm": fwhm,
                    "srf_source": np.array(["catalog"] * wavelengths.size),
                    "valid_mask": band_mask if band_mask is not None else np.ones_like(wavelengths, dtype=bool),
                }
            quality_masks = {name: mask[y, x, :] for name, mask in quality_base.items()}
            yield Sample(
                spectrum=spec,
                sensor_id="enmap",
                band_meta=band_meta,
                quality_masks=quality_masks,
                srf_matrix=srfs_for_sensor,
                ancillary={"source_path": str(path_or_pair), "y": int(y), "x": int(x)},
            )


def load_enmap_scene(path_or_pair: str | tuple[str, str]) -> List[Sample]:
    """Load an EnMAP scene into memory."""

    return list(iter_enmap_pixels(path_or_pair))
