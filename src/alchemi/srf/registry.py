from __future__ import annotations

"""Process-wide SRF registry backed by canonical :class:`SensorSRF` objects."""

import warnings
from pathlib import Path
from typing import Iterable

import numpy as np

from ..registry import srfs
from ..spectral.srf import SensorSRF, SRFProvenance
from ..types import SRFMatrix as LegacySRFMatrix, SRFMatrix


def _to_dense_matrix(
    legacy: LegacySRFMatrix,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a legacy :class:`~alchemi.types.SRFMatrix` to a dense matrix.

    Parameters
    ----------
    legacy:
        Legacy SRF container using per-band wavelength supports.
    grid:
        Optional wavelength grid in nanometres. When omitted, the union of band
        supports is used.
    """
    base_grid = (
        np.unique(np.concatenate([np.asarray(b, dtype=np.float64) for b in legacy.bands_nm]))
        if grid is None
        else np.asarray(grid, dtype=np.float64)
    )
    base_grid.setflags(write=False)

    matrix = np.zeros((len(legacy.bands_nm), base_grid.shape[0]), dtype=np.float64)

    for idx, (nm, resp) in enumerate(zip(legacy.bands_nm, legacy.bands_resp, strict=True)):
        nm_arr = np.asarray(nm, dtype=np.float64)
        resp_arr = np.asarray(resp, dtype=np.float64)

        matrix[idx] = np.interp(base_grid, nm_arr, resp_arr, left=0.0, right=0.0)

        area = float(np.trapz(matrix[idx], x=base_grid))
        if area > 0.0:
            matrix[idx] /= area

    return base_grid, matrix


def sensor_srf_from_legacy(
    legacy: LegacySRFMatrix,
    *,
    grid: np.ndarray | None = None,
    provenance: SRFProvenance = SRFProvenance.OFFICIAL,
    band_widths_nm: Iterable[float] | None = None,
    valid_mask: np.ndarray | None = None,
) -> SensorSRF:
    """Wrap a legacy SRF matrix into the canonical :class:`SensorSRF`."""

    wavelength_grid, matrix = _to_dense_matrix(legacy, grid)

    if band_widths_nm is not None:
        widths = np.asarray(list(band_widths_nm), dtype=np.float64)
    else:
        widths = np.asarray(
            [np.ptp(np.asarray(b, dtype=np.float64)) for b in legacy.bands_nm],
            dtype=np.float64,
        )

    return SensorSRF(
        sensor_id=legacy.sensor,
        wavelength_grid_nm=wavelength_grid,
        srfs=matrix,
        band_centers_nm=np.asarray(legacy.centers_nm, dtype=np.float64),
        band_widths_nm=widths,
        provenance=provenance,
        valid_mask=valid_mask,
        meta={
            "version": getattr(legacy, "version", None),
            "cache_key": getattr(legacy, "cache_key", None),
        },
    )


class SRFRegistry:
    """Compatibility shim that proxies to :mod:`alchemi.registry.srfs`.

    New code should call :func:`alchemi.registry.srfs.get_srf` directly. This
    wrapper preserves the legacy interface used by older adapters while routing
    all lookups through the canonical SRF registry.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        warnings.warn(
            "alchemi.srf.registry.SRFRegistry is deprecated; use"
            " alchemi.registry.srfs.get_srf instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self._root = Path(root) if root is not None else None
        self._overrides: dict[str, SRFMatrix] = {}

    def register(self, sensor_srf: SensorSRF | SRFMatrix) -> None:
        key = sensor_srf.sensor_id.lower()
        if isinstance(sensor_srf, SensorSRF):
            # Convert to the canonical matrix representation for downstream use.
            matrix = sensor_srf.to_matrix()
        else:
            matrix = sensor_srf
        self._overrides[key] = matrix

    def has(self, sensor_id: str) -> bool:
        return sensor_id.lower() in self._overrides

    def get(self, sensor_id: str) -> SRFMatrix | None:
        key = sensor_id.lower()
        if key in self._overrides:
            return self._overrides[key]
        try:
            return srfs.get_srf(sensor_id, base_path=self._root)
        except FileNotFoundError:
            return None

    def require(self, sensor_id: str) -> SRFMatrix:
        srf = self.get(sensor_id)
        if srf is None:  # pragma: no cover - defensive guard
            raise KeyError(f"No SRF registered for sensor_id={sensor_id!r}")
        return srf

    def all_sensors(self) -> list[str]:
        sensors = set(self._overrides.keys())
        if self._root is None:
            try:
                sensors.update(path.stem.split("_srfs")[0] for path in Path(srfs._SRF_ROOT).glob("*_srfs.*"))
            except Exception:  # pragma: no cover - best effort
                pass
        return sorted(sensors)


GLOBAL_SRF_REGISTRY = SRFRegistry()


def get_srf(sensor_id: str, **_: object) -> SRFMatrix | None:
    """Backwards-compatible helper that proxies the global registry."""
    return GLOBAL_SRF_REGISTRY.get(sensor_id)


def register_sensor_srf(sensor_srf: SensorSRF) -> None:
    GLOBAL_SRF_REGISTRY.register(sensor_srf)


def register_virtual_sensor(sensor_srf: SensorSRF, *, registry: SRFRegistry | None = None) -> None:
    """Compatibility alias for registering synthetic/virtual sensors."""
    target = registry or GLOBAL_SRF_REGISTRY
    target.register(sensor_srf)


__all__ = [
    "GLOBAL_SRF_REGISTRY",
    "SRFRegistry",
    "SensorSRF",
    "SRFProvenance",
    "get_srf",
    "register_sensor_srf",
    "register_virtual_sensor",
    "sensor_srf_from_legacy",
]
