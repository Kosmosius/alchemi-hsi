"""Utilities for loading and normalizing EMIT spectral response functions (SRFs)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from importlib import resources
from typing import overload

import numpy as np
from numpy.typing import NDArray

from alchemi.types import SRFMatrix

_RESOURCE_PACKAGE = __package__ + ".data"
_RESOURCE_NAME = "emit_srf_v01.json"
_SENSOR = "EMIT"
_VERSION = "v01"


class _EmitSRFArchive(Sequence[NDArray[np.float64]]):
    """Container for EMIT SRF response curves on the native wavelength grid."""

    def __init__(
        self,
        native_nm: np.ndarray,
        centers_nm: np.ndarray,
        responses: np.ndarray,
    ) -> None:
        self.native_nm = np.asarray(native_nm, dtype=np.float64)
        self.centers_nm = np.asarray(centers_nm, dtype=np.float64)
        self.responses = np.asarray(responses, dtype=np.float64)
        if self.native_nm.ndim != 1 or not np.all(np.diff(self.native_nm) > 0):
            raise ValueError("Native wavelength grid must be increasing and one-dimensional")
        if self.responses.ndim != 2:
            raise ValueError("Responses must be a 2-D array of shape (bands, samples)")
        if self.responses.shape[1] != self.native_nm.shape[0]:
            raise ValueError("Responses must align with the native wavelength grid")
        if self.responses.shape[0] != self.centers_nm.shape[0]:
            raise ValueError("Response count must match number of band centers")

    def __len__(self) -> int:
        return int(self.responses.shape[0])

    @overload
    def __getitem__(self, idx: int) -> np.ndarray: ...

    @overload
    def __getitem__(self, idx: slice) -> Sequence[np.ndarray]: ...

    def __getitem__(self, idx: int | slice) -> NDArray[np.float64] | Sequence[NDArray[np.float64]]:
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.__len__())
            return [
                np.asarray(self.responses[i], dtype=np.float64) for i in range(start, stop, step)
            ]
        return np.asarray(self.responses[idx], dtype=np.float64)


def _load_emit_archive() -> _EmitSRFArchive:
    """Load the packaged EMIT SRF archive."""

    data_path = resources.files(_RESOURCE_PACKAGE).joinpath(_RESOURCE_NAME)
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    native = np.asarray(raw["native"], dtype=np.float64)
    centers = np.asarray(raw["centers"], dtype=np.float64)
    responses = np.asarray(raw["responses"], dtype=np.float64)
    return _EmitSRFArchive(native, centers, responses)


def _resample_to_grid(
    archive: _EmitSRFArchive, highres_wl_nm: np.ndarray
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Resample EMIT SRF responses to the provided high-resolution wavelength grid."""

    highres = np.asarray(highres_wl_nm, dtype=np.float64)
    if highres.ndim != 1 or not np.all(np.diff(highres) > 0):
        raise ValueError("High-resolution wavelength grid must be strictly increasing")

    nm_bands: list[NDArray[np.float64]] = []
    resp_bands: list[NDArray[np.float64]] = []
    for resp in archive:
        sampled = np.interp(highres, archive.native_nm, resp, left=0.0, right=0.0)
        mask = sampled > 1e-8
        if mask.sum() < 2:
            # Guarantee a minimally sized support to keep integration stable.
            mask = (highres >= archive.native_nm[0]) & (highres <= archive.native_nm[-1])
        nm_band = highres[mask]
        resp_band = sampled[mask]
        if nm_band.size < 2:
            # Ensure at least a small baseline if interpolation produced degeneracy.
            indices = np.clip(
                np.searchsorted(highres, archive.native_nm[[0, -1]]),
                0,
                highres.size - 1,
            )
            nm_band = highres[indices]
            resp_band = sampled[indices]
        nm_bands.append(np.asarray(nm_band, dtype=np.float64))
        resp_bands.append(np.asarray(resp_band, dtype=np.float64))
    return nm_bands, resp_bands


def _compute_cache_key(
    centers: np.ndarray, nm: Iterable[np.ndarray], resp: Iterable[np.ndarray]
) -> str:
    hasher = hashlib.sha1()
    hasher.update(_SENSOR.lower().encode("utf-8"))
    hasher.update(_VERSION.encode("utf-8"))
    hasher.update(np.asarray(centers, dtype=np.float64).tobytes())
    for band_nm, band_resp in zip(nm, resp, strict=True):
        hasher.update(np.asarray(band_nm, dtype=np.float64).tobytes())
        hasher.update(np.asarray(band_resp, dtype=np.float64).tobytes())
    digest = hasher.hexdigest()[:12]
    return f"{_SENSOR.lower()}:{_VERSION}:{digest}"


def emit_srf_matrix(highres_wl_nm: np.ndarray) -> SRFMatrix:
    """Return the EMIT SRF matrix resampled onto *highres_wl_nm*.

    The resulting :class:`~alchemi.types.SRFMatrix` is normalized such that each
    band integrates to 1.0 when evaluated with the trapezoidal rule and includes a
    cache key derived from the resampled responses.
    """

    archive = _load_emit_archive()
    nm_bands, resp_bands = _resample_to_grid(archive, highres_wl_nm)
    srf = SRFMatrix(_SENSOR, archive.centers_nm, nm_bands, resp_bands, version=_VERSION)
    srf = srf.normalize_trapz()
    srf.cache_key = _compute_cache_key(srf.centers_nm, srf.bands_nm, srf.bands_resp)
    return srf


__all__ = ["emit_srf_matrix"]
