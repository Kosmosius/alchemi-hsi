from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from alchemi.registry.srfs import get_srf
from alchemi.types import QuantityKind

from .cube import Cube


def validate_dataset(cfg: dict[str, Any]) -> None:
    dp = cfg.get("data", {})
    for key in ["paths", "wavelengths"]:
        if key not in dp:
            raise ValueError(f"Missing data.{key} in config")


def validate_srf_dir(srf_root: str | os.PathLike[str]) -> None:
    root = Path(srf_root)
    if not root.exists():
        raise FileNotFoundError(f"SRF root not found: {root}")
    for p in root.glob("*.json"):
        obj = json.loads(p.read_text())
        centers = np.array(obj["centers_nm"], dtype=float)
        if not np.all(np.diff(centers) > 0):
            raise ValueError(f"Non-monotonic centers in {p.name}")
        for b in obj["bands"]:
            nm = np.array(b["nm"], dtype=float)
            resp = np.array(b["resp"], dtype=float)
            if nm.shape != resp.shape or (resp < 0).any():
                raise ValueError(f"Invalid SRF row in {p.name}")


_logger = logging.getLogger(__name__)


def _resolve_sensor_id(cube: Cube, sensor_id: str | None) -> str | None:
    if sensor_id:
        return str(sensor_id)

    for key in (cube.sensor, cube.srf_id, cube.attrs.get("sensor")):
        if key is not None:
            return str(key)
    return None


def _sample_finite(data: np.ndarray) -> np.ndarray:
    flat = np.asarray(data).ravel()
    mask = np.isfinite(flat)
    return flat[mask]


def check_cube_health(
    cube: Cube, *, sensor_id: str | None = None, logger: logging.Logger | None = None
) -> None:
    """Perform lightweight sanity checks on a :class:`Cube`.

    Structural issues raise :class:`ValueError`; value-range problems are logged so
    that ingest callers can decide how to handle them downstream.
    """

    log = logger or _logger
    resolved_sensor = _resolve_sensor_id(cube, sensor_id)

    # Spectral axis sanity checks
    axis = np.asarray(cube.axis)
    if axis.ndim != 1:
        raise ValueError("Cube axis must be one-dimensional")
    if axis.shape[0] != cube.data.shape[-1]:
        raise ValueError("Cube axis length must match spectral dimension")
    if not np.all(np.diff(axis) > 0):
        raise ValueError("Cube axis must be strictly monotonic increasing")

    # Band mask compatibility
    if cube.band_mask is not None and cube.band_mask.shape[0] != axis.shape[0]:
        raise ValueError("band_mask length must match spectral axis")

    finite_values = _sample_finite(cube.data)
    if finite_values.size == 0:
        log.warning("Cube contains no finite values", extra={"sensor": resolved_sensor})
        return

    min_val = float(np.nanmin(finite_values))
    max_val = float(np.nanmax(finite_values))

    if cube.value_kind == QuantityKind.RADIANCE:
        if min_val < -1e-6:
            log.error(
                "Radiance cube has negative values",
                extra={"min": min_val, "sensor": resolved_sensor},
            )
        if max_val > 1e5:
            log.warning(
                "Radiance cube has very large values",
                extra={"max": max_val, "sensor": resolved_sensor},
            )
    elif cube.value_kind in {
        QuantityKind.REFLECTANCE,
        QuantityKind.SURFACE_REFLECTANCE,
        QuantityKind.TOA_REFLECTANCE,
    }:
        q99 = float(np.quantile(finite_values, 0.99))
        if q99 < 0 or q99 > 1.5:
            log.warning(
                "Reflectance cube 99th percentile outside [0, 1.5]",
                extra={"q99": q99, "sensor": resolved_sensor},
            )
    elif cube.value_kind == QuantityKind.BRIGHTNESS_T:
        if min_val < 100 or max_val > 500:
            log.warning(
                "Brightness temperature cube outside expected range",
                extra={"min": min_val, "max": max_val, "sensor": resolved_sensor},
            )

    if resolved_sensor:
        try:
            srf = get_srf(resolved_sensor)
        except FileNotFoundError:
            return
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("SRF lookup failed", exc_info=exc)
            return

        srf_band_count = int(getattr(srf, "band_count", len(getattr(srf, "centers_nm", []))))
        cube_band_count = cube.data.shape[-1]
        if srf_band_count and srf_band_count != cube_band_count:
            log.warning(
                "SRF band count mismatch with cube",
                extra={
                    "srf_bands": srf_band_count,
                    "cube_bands": cube_band_count,
                    "sensor": resolved_sensor,
                },
            )
