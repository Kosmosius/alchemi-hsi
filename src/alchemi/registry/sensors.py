from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, Tuple

import numpy as np


@dataclass
class SensorSpec:
    sensor_id: str
    expected_band_count: int
    wavelength_range_nm: Tuple[float, float]
    band_centers_nm: np.ndarray
    band_widths_nm: np.ndarray
    srf_source: Literal["official", "gaussian", "none", "synthetic"]
    bad_band_mask: np.ndarray | None = None
    absorption_windows_nm: Sequence[tuple[float, float]] | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        centers = np.asarray(self.band_centers_nm, dtype=np.float64)
        widths = np.asarray(self.band_widths_nm, dtype=np.float64)

        if centers.ndim != 1:
            raise ValueError("band_centers_nm must be 1-D")
        if widths.ndim != 1:
            raise ValueError("band_widths_nm must be 1-D")
        if centers.shape[0] != widths.shape[0]:
            raise ValueError("band_centers_nm and band_widths_nm must have the same length")
        if centers.shape[0] != self.expected_band_count:
            raise ValueError("expected_band_count does not match band definitions")

        self.band_centers_nm = centers
        self.band_widths_nm = widths

        if self.bad_band_mask is not None:
            mask = np.asarray(self.bad_band_mask, dtype=bool)
            if mask.shape != centers.shape:
                raise ValueError("bad_band_mask shape must match band_centers_nm")
            self.bad_band_mask = mask


class SensorRegistry:
    """Lightweight registry of sensor specifications.

    Specifications are stored by lowercase sensor identifier to make lookups
    case-insensitive. The registry is intentionally minimal and keeps
    everything in-memory.
    """

    def __init__(self) -> None:
        self._specs: dict[str, SensorSpec] = {}

    def register_sensor(self, spec: SensorSpec) -> None:
        self._specs[spec.sensor_id.lower()] = spec

    def get_sensor(self, sensor_id: str) -> SensorSpec:
        key = sensor_id.lower()
        try:
            return self._specs[key]
        except KeyError as exc:  # pragma: no cover - defensive guard
            msg = f"Unknown sensor_id: {sensor_id!r}"
            raise KeyError(msg) from exc

    def list_sensors(self) -> list[str]:
        return sorted(self._specs)


def _seed_registry() -> SensorRegistry:
    registry = SensorRegistry()

    emit_centers = np.linspace(380.0, 2500.0, 285, dtype=np.float64)
    emit_widths = np.full_like(emit_centers, 10.0, dtype=np.float64)
    registry.register_sensor(
        SensorSpec(
            "emit",
            expected_band_count=emit_centers.size,
            wavelength_range_nm=(380.0, 2500.0),
            band_centers_nm=emit_centers,
            band_widths_nm=emit_widths,
            srf_source="official",
            absorption_windows_nm=[(1350.0, 1450.0), (1800.0, 1950.0)],
            notes="TODO: replace placeholder band widths with values from official EMIT SRFs.",
        )
    )

    enmap_centers = np.linspace(420.0, 2450.0, 242, dtype=np.float64)
    enmap_widths = np.full_like(enmap_centers, 12.0, dtype=np.float64)
    registry.register_sensor(
        SensorSpec(
            "enmap",
            expected_band_count=enmap_centers.size,
            wavelength_range_nm=(420.0, 2450.0),
            band_centers_nm=enmap_centers,
            band_widths_nm=enmap_widths,
            srf_source="official",
            absorption_windows_nm=[(1330.0, 1450.0), (1800.0, 2000.0)],
            notes="TODO: update with EnMAP bandpass table from mission documentation.",
        )
    )

    aviris_ng_centers = np.linspace(380.0, 2500.0, 425, dtype=np.float64)
    aviris_ng_widths = np.full_like(aviris_ng_centers, 8.0, dtype=np.float64)
    registry.register_sensor(
        SensorSpec(
            "aviris-ng",
            expected_band_count=aviris_ng_centers.size,
            wavelength_range_nm=(380.0, 2500.0),
            band_centers_nm=aviris_ng_centers,
            band_widths_nm=aviris_ng_widths,
            srf_source="official",
            absorption_windows_nm=[(1340.0, 1455.0), (1790.0, 1950.0)],
            notes="TODO: substitute official AVIRIS-NG SRF measurements once ingested.",
        )
    )

    hytes_centers = np.linspace(7600.0, 13200.0, 256, dtype=np.float64)
    hytes_widths = np.full_like(hytes_centers, 44.0, dtype=np.float64)
    registry.register_sensor(
        SensorSpec(
            "hytes",
            expected_band_count=hytes_centers.size,
            wavelength_range_nm=(7600.0, 13200.0),
            band_centers_nm=hytes_centers,
            band_widths_nm=hytes_widths,
            srf_source="gaussian",
            absorption_windows_nm=[],
            notes="TODO: refine HyTES LWIR SRFs when calibrated files are available.",
        )
    )

    return registry


DEFAULT_SENSOR_REGISTRY = _seed_registry()


__all__ = ["SensorSpec", "SensorRegistry", "DEFAULT_SENSOR_REGISTRY"]
