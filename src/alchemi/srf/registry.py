from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..types import SRFMatrix
from ..utils.logging import get_logger
from .emit import emit_srf_matrix
from .mako import build_mako_srf_from_header, mako_lwir_grid_nm

_LOG = get_logger(__name__)

_DEFAULT_MAKO_WAVELENGTHS_NM = np.linspace(7600.0, 13200.0, 128, dtype=np.float64)
_DEFAULT_MAKO_VERSION = "comex-l2s-gaussian-v1"
_DEFAULT_MAKO_FWHM = 44.0
_DEFAULT_EMIT_GRID = np.linspace(380.0, 2500.0, 2000, dtype=np.float64)


class SRFRegistry:
    def __init__(self, root: str | Path = "data/srf"):
        self.root = Path(root)
        self._cache: dict[str, SRFMatrix] = {}

    def _hash(self, payload: str) -> str:
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def get(self, sensor: str) -> SRFMatrix:
        k = sensor.lower()
        if k in self._cache:
            return self._cache[k]
        path = self.root / f"{k}.json"
        if not path.exists():
            raise FileNotFoundError(f"SRF file not found: {path}")
        obj = json.loads(path.read_text())
        centers = np.array(obj["centers_nm"], dtype=np.float64)
        bands_nm = [np.array(b["nm"], dtype=np.float64) for b in obj["bands"]]
        bands_resp = [np.array(b["resp"], dtype=np.float64) for b in obj["bands"]]
        srf = SRFMatrix(
            obj["sensor"],
            centers,
            bands_nm,
            bands_resp,
            version=obj.get("version", "v1"),
            cache_key=obj.get("cache_key"),
        )
        srf = srf.normalize_trapz()
        if not srf.cache_key:
            srf.cache_key = self._hash(path.read_text()[:2048])
        self._cache[k] = srf
        _LOG.info("Loaded SRF for %s (%d bands)", k, len(centers))
        return srf


def _mako_builder(
    *,
    version: str | None,
    wavelengths_nm: np.ndarray | None,
    fwhm_nm: float | None,
) -> tuple[SRFMatrix, np.ndarray]:
    centers = (
        _DEFAULT_MAKO_WAVELENGTHS_NM
        if wavelengths_nm is None
        else np.asarray(wavelengths_nm, dtype=np.float64)
    )
    fwhm = float(_DEFAULT_MAKO_FWHM if fwhm_nm is None else fwhm_nm)
    srf = build_mako_srf_from_header(centers, fwhm_nm=fwhm)
    srf.version = version or _DEFAULT_MAKO_VERSION
    grid = mako_lwir_grid_nm()
    return srf, grid


def _emit_builder(
    *,
    version: str | None,
    wavelengths_nm: np.ndarray | None,
    fwhm_nm: float | None,
) -> tuple[SRFMatrix, np.ndarray]:
    if fwhm_nm is not None:
        _LOG.warning("Ignoring FWHM override for EMIT SRFs")
    grid = (
        _DEFAULT_EMIT_GRID
        if wavelengths_nm is None
        else np.asarray(wavelengths_nm, dtype=np.float64)
    )
    srf = emit_srf_matrix(grid)
    if version is not None:
        srf.version = version
    return srf, grid


_BUILTIN_BUILDERS: dict[str, Callable[..., tuple[SRFMatrix, np.ndarray]]] = {
    "mako": _mako_builder,
    "emit": _emit_builder,
}


def get_srf(
    sensor: str,
    *,
    version: str | None = None,
    wavelengths_nm: np.ndarray | None = None,
    fwhm_nm: float | None = None,
) -> tuple[SRFMatrix, np.ndarray]:
    """Return a builtin SRF matrix together with its canonical wavelength grid."""

    key = sensor.lower()
    try:
        builder = _BUILTIN_BUILDERS[key]
    except KeyError as exc:  # pragma: no cover - defensive guard
        msg = f"Unsupported builtin SRF sensor: {sensor!r}"
        raise ValueError(msg) from exc
    return builder(version=version, wavelengths_nm=wavelengths_nm, fwhm_nm=fwhm_nm)


__all__ = ["SRFRegistry", "get_srf"]
