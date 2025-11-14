from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from alchemi.types import SRFMatrix

_ENMAP_SENSOR = "enmap"
_ENMAP_VERSION = "v1"


def _band_centers_nm() -> np.ndarray:
    """Approximate EnMAP band centers (nm).

    The VNIR module spans roughly 420-999 nm with 95 samples while the SWIR
    module covers 1001-2450 nm with 131 samples. The slight overlap is used by
    downstream processing to stitch the two spectrometers.
    """

    vnir = np.linspace(420.0, 999.0, 95, dtype=np.float64)
    swir = np.linspace(1001.0, 2450.0, 131, dtype=np.float64)
    return np.concatenate([vnir, swir])


def _band_grid(center: float, *, points: int, width: float) -> np.ndarray:
    half = width / 2.0
    return np.linspace(center - half, center + half, points, dtype=np.float64)


def _band_response(nm: np.ndarray, center: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((nm - center) / sigma) ** 2)


def _synthesize_srf() -> SRFMatrix:
    centers = _band_centers_nm()
    bands_nm: list[np.ndarray] = []
    bands_resp: list[np.ndarray] = []
    for c in centers:
        if c <= 999.0:  # VNIR
            grid = _band_grid(c, points=9, width=6.0)
            resp = _band_response(grid, c, sigma=1.4)
        else:  # SWIR
            grid = _band_grid(c, points=11, width=10.0)
            resp = _band_response(grid, c, sigma=2.5)
        bands_nm.append(grid)
        bands_resp.append(resp)
    return SRFMatrix(_ENMAP_SENSOR, centers, bands_nm, bands_resp, version=_ENMAP_VERSION)


def _cache_key_payload(srf: SRFMatrix) -> Iterable[bytes]:
    yield np.asarray(srf.centers_nm, dtype=np.float64).tobytes()
    for nm, resp in zip(srf.bands_nm, srf.bands_resp, strict=True):
        yield np.asarray(nm, dtype=np.float64).tobytes()
        yield np.asarray(resp, dtype=np.float64).tobytes()


def _compute_cache_key(srf: SRFMatrix) -> str:
    h = hashlib.sha1()
    for payload in _cache_key_payload(srf):
        h.update(payload)
    return f"{srf.sensor}:{srf.version}:{h.hexdigest()[:12]}"


def _serialize_srf(srf: SRFMatrix) -> dict:
    return {
        "sensor": srf.sensor,
        "version": srf.version,
        "cache_key": srf.cache_key,
        "centers_nm": srf.centers_nm.tolist(),
        "bands": [
            {"nm": nm.tolist(), "resp": resp.tolist()}
            for nm, resp in zip(srf.bands_nm, srf.bands_resp, strict=True)
        ],
    }


def _persist(cache_dir: Path, srf: SRFMatrix) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = _serialize_srf(srf)
    path = cache_dir / f"{srf.sensor}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def enmap_srf_matrix(*, cache_dir: str | Path = "data/srf", force: bool = False) -> SRFMatrix:
    """Build and cache an approximate EnMAP SRF matrix.

    Parameters
    ----------
    cache_dir:
        Directory where the normalized SRF JSON representation should be
        written. A temporary directory can be supplied by the caller for test
        isolation.
    force:
        When ``True`` the JSON cache will be overwritten even if the existing
        cache key matches the synthesized SRF.
    """

    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{_ENMAP_SENSOR}.json"

    srf = _synthesize_srf().normalize_trapz()
    srf.cache_key = _compute_cache_key(srf)

    if cache_path.exists() and not force:
        try:
            existing = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            existing = None
        if isinstance(existing, dict) and existing.get("cache_key") == srf.cache_key:
            # Reuse the cached representation.
            bands_nm = [np.asarray(b["nm"], dtype=np.float64) for b in existing["bands"]]
            bands_resp = [np.asarray(b["resp"], dtype=np.float64) for b in existing["bands"]]
            cached = SRFMatrix(
                existing.get("sensor", _ENMAP_SENSOR),
                np.asarray(existing.get("centers_nm", srf.centers_nm), dtype=np.float64),
                bands_nm,
                bands_resp,
                version=existing.get("version", _ENMAP_VERSION),
                cache_key=existing.get("cache_key", srf.cache_key),
            )
            return cached.normalize_trapz()

    _persist(cache_dir, srf)
    return srf
