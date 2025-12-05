"""Sample data structure tying together spectra and acquisition metadata.

Design details follow the "Data and metadata model" section of the ALCHEMI
specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .spectrum import Spectrum
from .srf import SRFMatrix


@dataclass
class ViewingGeometry:
    solar_zenith_deg: float
    solar_azimuth_deg: float
    view_zenith_deg: float
    view_azimuth_deg: float
    earth_sun_distance_au: float


@dataclass
class BandMetadata:
    center_nm: NDArray[np.floating]
    width_nm: NDArray[np.floating]
    srf_source: NDArray[np.object_] | NDArray[np.str_] | list[str]
    valid_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        self.center_nm = np.asarray(self.center_nm, dtype=float)
        self.width_nm = np.asarray(self.width_nm, dtype=float)
        self.srf_source = np.asarray(self.srf_source)
        self.valid_mask = np.asarray(self.valid_mask, dtype=bool)

    def validate_length(self, length: int) -> None:
        if self.center_nm.shape[0] != length:
            raise ValueError("center_nm length must match spectrum length")
        if self.width_nm.shape[0] != length:
            raise ValueError("width_nm length must match spectrum length")
        if self.srf_source.shape[0] != length:
            raise ValueError("srf_source length must match spectrum length")
        if self.valid_mask.shape[0] != length:
            raise ValueError("valid_mask length must match spectrum length")


@dataclass
class Sample:
    spectrum: Spectrum
    sensor_id: str
    acquisition_time: Optional[datetime] = None
    geolocation: Optional[Tuple[float, float, float]] = None
    viewing_geometry: ViewingGeometry | Mapping[str, float] | None = None
    band_meta: BandMetadata | Mapping[str, Any] | None = None
    srf_matrix: Optional[SRFMatrix] = None
    quality_masks: Dict[str, NDArray[np.bool_]] = field(default_factory=dict)
    ancillary: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.viewing_geometry, Mapping) and not isinstance(
            self.viewing_geometry, ViewingGeometry
        ):
            self.viewing_geometry = ViewingGeometry(**self.viewing_geometry)
        if isinstance(self.band_meta, Mapping) and not isinstance(self.band_meta, BandMetadata):
            self.band_meta = BandMetadata(**self.band_meta)
        self.quality_masks = dict(self.quality_masks)
        self.ancillary = dict(self.ancillary)
        self.validate()

    def validate(self) -> None:
        if hasattr(self.spectrum, "validate"):
            self.spectrum.validate()
        length = self.spectrum.band_count

        if self.band_meta is not None:
            if not isinstance(self.band_meta, BandMetadata):
                raise TypeError("band_meta must be a BandMetadata instance")
            self.band_meta.validate_length(length)

        if self.srf_matrix is not None:
            if self.srf_matrix.matrix.shape[1] != length:
                raise ValueError("SRF matrix wavelength axis must match spectrum length")
            if self.band_meta is not None and self.srf_matrix.matrix.shape[0] != self.band_meta.center_nm.shape[0]:
                raise ValueError("SRF matrix band count must match band metadata")

        for name, mask in self.quality_masks.items():
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape != self.spectrum.values.shape:
                raise ValueError(f"Quality mask '{name}' must match spectrum shape")
            self.quality_masks[name] = mask_arr

        if self.geolocation is not None and len(self.geolocation) != 3:
            raise ValueError("geolocation must be a (lat, lon, elev) tuple")

        if self.viewing_geometry is not None and not isinstance(self.viewing_geometry, ViewingGeometry):
            raise TypeError("viewing_geometry must be a ViewingGeometry instance")

    def to_chip(self) -> NDArray[np.floating]:
        """Return the spectrum as a (1, 1, L) chip suitable for cube insertion."""

        return self.spectrum.values[np.newaxis, np.newaxis, :]

    @classmethod
    def from_chip(
        cls,
        cube: NDArray[np.floating],
        wavelength_nm: NDArray[np.floating],
        *,
        sensor_id: str,
        kind: str,
        row: int = 0,
        col: int = 0,
        **kwargs: Any,
    ) -> "Sample":
        cube = np.asarray(cube, dtype=float)
        if cube.ndim != 3:
            raise ValueError("cube must be 3-D (H, W, L)")
        if cube.shape[2] != np.asarray(wavelength_nm).shape[0]:
            raise ValueError("cube spectral dimension must match wavelength grid length")
        values = cube[row, col, :]
        spectrum = Spectrum(wavelength_nm=np.asarray(wavelength_nm, dtype=float), values=values, kind=kind)
        return cls(spectrum=spectrum, sensor_id=sensor_id, **kwargs)
