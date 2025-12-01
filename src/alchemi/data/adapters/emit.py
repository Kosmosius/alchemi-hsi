"""Stub adapter for EMIT scenes returning :class:`~alchemi.spectral.sample.Sample` objects.

The implementation focuses on the expected data flow (unit conversions,
wavelength handling, quality masks, and SRF lookup) while leaving TODOs for the
mission-specific parsing logic.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, List

import numpy as np
import xarray as xr

from alchemi.registry import srfs
from alchemi.spectral import Sample, Spectrum

from ..io import load_emit_l1b

__all__ = ["load_emit_scene", "iter_emit_pixels"]


def _normalize_radiance_units(radiance: xr.DataArray) -> np.ndarray:
    """Convert radiance to W/m^2/sr/nm when possible.

    EMIT provides radiance in W/m^2/sr/µm. We rescale to per-nm to match the
    spectral model used elsewhere in the codebase. This function is intentionally
    lightweight; real implementations should inspect official metadata.
    """

    values = np.asarray(radiance.values, dtype=np.float64)
    units = str(radiance.attrs.get("units", "")).lower()
    if "um" in units or "µm" in units:
        return values / 1_000.0
    return values


def iter_emit_pixels(path: str, *, include_quality: bool = True) -> Iterable[Sample]:
    """Iterate over pixels in an EMIT scene.

    Parameters
    ----------
    path:
        Path to an EMIT L1B product supported by :func:`alchemi.data.io.load_emit_l1b`.
    include_quality:
        When ``True`` include QA masks packaged alongside the dataset if present.
    """

    ds = load_emit_l1b(path)
    wavelengths = np.asarray(ds["wavelength_nm"].values, dtype=np.float64)
    srfs_for_sensor = None
    try:
        srfs_for_sensor = srfs.get_srf("emit")
    except Exception:
        # TODO: handle SRF lookup failures once registry paths are configured in fixtures.
        srfs_for_sensor = None

    band_mask = None
    if include_quality and "band_mask" in ds:
        band_mask = np.asarray(ds["band_mask"].values, dtype=bool)

    radiance = ds["radiance"]
    scaled = _normalize_radiance_units(radiance)

    quality_base: Dict[str, np.ndarray] = {}
    if band_mask is not None:
        quality_base["band_mask"] = np.broadcast_to(band_mask, radiance.shape)

    # TODO: incorporate mission QA layers (clouds, glint, saturation) once the
    # product specification is wired in.
    for y in range(scaled.shape[0]):
        for x in range(scaled.shape[1]):
            values = scaled[y, x, :]
            spectrum = Spectrum(wavelength_nm=wavelengths, values=values, kind="radiance")
            quality_masks = {name: mask[y, x, :] for name, mask in quality_base.items()}
            yield Sample(
                spectrum=spectrum,
                sensor_id="emit",
                quality_masks=quality_masks,
                srf_matrix=srfs_for_sensor,
                ancillary={"source_path": path, "y": int(y), "x": int(x)},
            )


def load_emit_scene(path: str, *, include_quality: bool = True) -> List[Sample]:
    """Load an EMIT scene into a list of :class:`Sample` objects.

    This helper materialises the iterator returned by :func:`iter_emit_pixels` for
    convenience in small benchmarks. Large scenes should prefer streaming via the
    iterator to avoid memory pressure.
    """

    return list(iter_emit_pixels(path, include_quality=include_quality))


# Legacy exports maintained for compatibility with earlier adapters. These call
# through to the new iterator implementation.
def load_emit_pixel(path: str, y: int, x: int, **_: Any) -> Spectrum:  # pragma: no cover - thin wrapper
    ds = load_emit_l1b(path)
    wavelengths = np.asarray(ds["wavelength_nm"].values, dtype=np.float64)
    radiance = ds["radiance"].isel(y=y, x=x)
    values = _normalize_radiance_units(radiance)
    return Spectrum(wavelength_nm=wavelengths, values=values, kind="radiance")
