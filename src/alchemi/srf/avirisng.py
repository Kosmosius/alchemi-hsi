"""AVIRIS-NG sensor response function registry."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from alchemi.types import SRFMatrix

__all__ = ["avirisng_srf_matrix", "avirisng_bad_band_mask"]

_SENSOR_ID = "avirisng"
_SRF_VERSION = "v1"
_CACHE_GLOB = f"{_SENSOR_ID}_{_SRF_VERSION}_*.json"
_BAD_BAND_WINDOWS_NM: Sequence[tuple[float, float]] = (
    (1340.0, 1460.0),
    (1790.0, 1960.0),
)


def avirisng_srf_matrix(cache_dir: str | Path | None = None) -> SRFMatrix:
    """Return the normalized AVIRIS-NG SRF matrix.

    Parameters
    ----------
    cache_dir:
        Optional directory on disk where the normalized SRF definition should be cached.
        When provided, the function will attempt to read an existing cache entry before
        falling back to the built-in generator. Any newly generated SRF matrix will also
        be serialized to this location.
    """

    token = ""
    if cache_dir is not None:
        token = str(Path(cache_dir).expanduser().resolve())
    matrix = _load_matrix(token)
    return _clone_matrix(matrix)


def avirisng_bad_band_mask(wavelengths_nm: np.ndarray) -> np.ndarray:
    """Mask bands that fall inside water-vapor and known bad-band windows."""

    mask = np.ones_like(wavelengths_nm, dtype=bool)
    for lo, hi in _BAD_BAND_WINDOWS_NM:
        mask &= ~((wavelengths_nm >= lo) & (wavelengths_nm <= hi))
    return mask


@lru_cache(maxsize=None)
def _load_matrix(cache_token: str) -> SRFMatrix:
    cache_dir = Path(cache_token) if cache_token else None
    matrix: SRFMatrix | None = None

    if cache_dir is not None:
        matrix = _load_cached(cache_dir)

    if matrix is None:
        matrix = _build_matrix()
        if cache_dir is not None:
            _write_cache(cache_dir, matrix)

    return matrix


def _build_matrix() -> SRFMatrix:
    centers = _avirisng_centers()
    bands_nm: list[np.ndarray] = []
    bands_resp: list[np.ndarray] = []

    for center in centers:
        nm_raw, resp_raw = _raw_band(center)
        nm_interp, resp_interp = _interpolate_band(nm_raw, resp_raw)
        bands_nm.append(nm_interp)
        bands_resp.append(resp_interp)

    matrix = SRFMatrix(_SENSOR_ID, centers, bands_nm, bands_resp, version=_SRF_VERSION)
    matrix = matrix.normalize_trapz()

    cache_hash = _compute_hash(matrix.centers_nm, matrix.bands_nm, matrix.bands_resp)
    matrix.cache_key = _format_cache_key(cache_hash)
    _attach_bad_band_mask(matrix)
    return matrix


def _avirisng_centers() -> np.ndarray:
    # AVIRIS-NG spans roughly 380–2510 nm across 425 contiguous bands.
    return np.linspace(380.0, 2510.0, 425, dtype=np.float64)


def _raw_band(center_nm: float) -> tuple[np.ndarray, np.ndarray]:
    # Construct a coarse Gaussian approximation for the native SRF samples.
    fwhm = _band_fwhm(center_nm)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    offsets = np.array([-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5], dtype=np.float64)
    nm = center_nm + offsets
    resp = np.exp(-0.5 * (offsets / sigma) ** 2)
    return nm, resp


def _band_fwhm(center_nm: float) -> float:
    if center_nm < 1000.0:
        return 6.0
    if center_nm < 1800.0:
        return 7.5
    return 9.0


def _interpolate_band(nm: np.ndarray, resp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    span = nm[-1] - nm[0]
    step = 0.5
    count = int(max(span / step, 1)) + 1
    grid = np.linspace(nm[0], nm[-1], count, dtype=np.float64)
    interp = np.interp(grid, nm, resp)
    return grid, interp


def _compute_hash(
    centers: np.ndarray, bands_nm: Iterable[np.ndarray], bands_resp: Iterable[np.ndarray]
) -> str:
    hasher = hashlib.sha1()
    hasher.update(np.asarray(centers, dtype=np.float64).tobytes())
    for nm, resp in zip(bands_nm, bands_resp):
        hasher.update(np.asarray(nm, dtype=np.float64).tobytes())
        hasher.update(np.asarray(resp, dtype=np.float64).tobytes())
    return hasher.hexdigest()[:12]


def _format_cache_key(cache_hash: str) -> str:
    return f"{_SENSOR_ID}:{_SRF_VERSION}:{cache_hash}"


def _extract_hash(cache_key: str | None) -> str:
    if not cache_key:
        return ""
    parts = cache_key.split(":")
    return parts[-1]


def _write_cache(cache_dir: Path, matrix: SRFMatrix) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_hash = _extract_hash(matrix.cache_key)
    path = cache_dir / f"{_SENSOR_ID}_{_SRF_VERSION}_{cache_hash}.json"
    payload = {
        "sensor": matrix.sensor,
        "version": matrix.version,
        "centers_nm": matrix.centers_nm.tolist(),
        "bands_nm": [band.tolist() for band in matrix.bands_nm],
        "bands_resp": [band.tolist() for band in matrix.bands_resp],
        "cache_key": matrix.cache_key,
    }
    path.write_text(json.dumps(payload))

    for other in cache_dir.glob(_CACHE_GLOB):
        if other != path:
            try:
                other.unlink()
            except FileNotFoundError:  # pragma: no cover - race condition safe-guard
                pass


def _load_cached(cache_dir: Path) -> SRFMatrix | None:
    candidates = sorted(cache_dir.glob(_CACHE_GLOB))
    if not candidates:
        return None

    path = candidates[-1]
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        return None

    if payload.get("sensor") != _SENSOR_ID:
        return None
    if payload.get("version", _SRF_VERSION) != _SRF_VERSION:
        return None

    matrix = SRFMatrix(
        _SENSOR_ID,
        np.asarray(payload["centers_nm"], dtype=np.float64),
        [np.asarray(band, dtype=np.float64) for band in payload["bands_nm"]],
        [np.asarray(band, dtype=np.float64) for band in payload["bands_resp"]],
        version=payload.get("version", _SRF_VERSION),
    )
    matrix = matrix.normalize_trapz()

    cache_hash = _compute_hash(matrix.centers_nm, matrix.bands_nm, matrix.bands_resp)
    matrix.cache_key = _format_cache_key(cache_hash)
    _attach_bad_band_mask(matrix)
    return matrix


def _attach_bad_band_mask(matrix: SRFMatrix) -> None:
    mask = avirisng_bad_band_mask(matrix.centers_nm)
    matrix.bad_band_mask = mask
    matrix.bad_band_windows_nm = tuple(_BAD_BAND_WINDOWS_NM)


def _clone_matrix(matrix: SRFMatrix) -> SRFMatrix:
    clone = SRFMatrix(
        matrix.sensor,
        np.asarray(matrix.centers_nm, dtype=np.float64).copy(),
        [np.asarray(band, dtype=np.float64).copy() for band in matrix.bands_nm],
        [np.asarray(band, dtype=np.float64).copy() for band in matrix.bands_resp],
        version=matrix.version,
        cache_key=matrix.cache_key,
    )
    if hasattr(matrix, "bad_band_mask"):
        clone.bad_band_mask = np.asarray(matrix.bad_band_mask, dtype=bool).copy()
    if hasattr(matrix, "bad_band_windows_nm"):
        clone.bad_band_windows_nm = tuple(matrix.bad_band_windows_nm)
    return clone

