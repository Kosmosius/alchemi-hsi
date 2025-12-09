"""Sample data structure tying together spectra and acquisition metadata.

The canonical :class:`Sample` couples a validated
:class:`~alchemi.spectral.spectrum.Spectrum` with sensor identifiers, geometry
metadata, spectral response functions, and mission-specific ancillary fields.
Downstream adapters, datasets, and alignment utilities should exclusively rely
on this type for per-pixel or lab spectra.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .spectrum import Spectrum
from .srf import SRFMatrix


@dataclass
class ViewingGeometry:
    """Solar/sensor viewing geometry in degrees with Earth-Sun distance."""

    solar_zenith_deg: float
    solar_azimuth_deg: float
    view_zenith_deg: float
    view_azimuth_deg: float
    earth_sun_distance_au: float


@dataclass
class GeoMeta:
    """Geolocation metadata expressed as latitude/longitude/elevation."""

    lat: float
    lon: float
    elev: float | None = None


@dataclass
class BandMetadata:
    """Per-band metadata covering centers, widths, validity, and SRF provenance."""

    center_nm: NDArray[np.floating]
    width_nm: NDArray[np.floating]
    valid_mask: NDArray[np.bool_]
    srf_source: NDArray[np.object_] | NDArray[np.str_] | list[str] | str = ""

    def __post_init__(self) -> None:
        self.center_nm = np.asarray(self.center_nm, dtype=float)
        self.width_nm = np.asarray(self.width_nm, dtype=float)
        self.srf_source = np.asarray(self.srf_source if self.srf_source is not None else "")
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
    """Canonical sample carrying a spectrum plus harmonised metadata."""

    spectrum: Spectrum
    sensor_id: str
    acquisition_time: Optional[datetime] = None
    geo: GeoMeta | Mapping[str, float] | Tuple[float, float, float] | None = None
    viewing_geometry: ViewingGeometry | Mapping[str, float] | None = None
    band_meta: BandMetadata | Mapping[str, Any] | None = None
    srf_matrix: Optional[SRFMatrix] = None
    quality_masks: Dict[str, NDArray[np.bool_]] = field(default_factory=dict)
    ancillary: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.spectrum = self._ensure_spectrum(self.spectrum)
        self.geo = self._normalize_geo(self.geo)
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
            if (
                self.band_meta is not None
                and self.srf_matrix.matrix.shape[0] != self.band_meta.center_nm.shape[0]
            ):
                raise ValueError("SRF matrix band count must match band metadata")

        for name, mask in self.quality_masks.items():
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.ndim != 1 or mask_arr.shape[0] != length:
                raise ValueError(f"Quality mask '{name}' must be 1-D with length {length}")
            self.quality_masks[name] = mask_arr

        if self.viewing_geometry is not None and not isinstance(
            self.viewing_geometry, ViewingGeometry
        ):
            raise TypeError("viewing_geometry must be a ViewingGeometry instance")

    def to_chip(self) -> NDArray[np.floating]:
        """Return the spectrum as a (1, 1, L) chip suitable for cube insertion."""

        return self.spectrum.values[np.newaxis, np.newaxis, :]

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the sample to a JSON-friendly dictionary."""

        def _array_to_list(arr: NDArray[np.floating] | NDArray[np.bool_] | None) -> Any:
            if arr is None:
                return None
            return arr.tolist()

        band_meta: Dict[str, Any] | None
        if isinstance(self.band_meta, BandMetadata):
            band_meta = {
                "center_nm": _array_to_list(self.band_meta.center_nm),
                "width_nm": _array_to_list(self.band_meta.width_nm),
                "valid_mask": _array_to_list(self.band_meta.valid_mask),
                "srf_source": _array_to_list(self.band_meta.srf_source),
            }
        else:
            band_meta = None

        return {
            "spectrum": {
                "wavelength_nm": self.spectrum.wavelength_nm.tolist(),
                "values": self.spectrum.values.tolist(),
                "kind": self.spectrum.kind,
            },
            "sensor_id": self.sensor_id,
            "acquisition_time": self.acquisition_time.isoformat()
            if isinstance(self.acquisition_time, datetime)
            else self.acquisition_time,
            "geo": asdict(self.geo) if isinstance(self.geo, GeoMeta) else self.geo,
            "viewing_geometry": asdict(self.viewing_geometry)
            if isinstance(self.viewing_geometry, ViewingGeometry)
            else self.viewing_geometry,
            "band_meta": band_meta,
            "srf_matrix": None
            if self.srf_matrix is None
            else {
                "wavelength_nm": _array_to_list(self.srf_matrix.wavelength_nm),
                "matrix": _array_to_list(self.srf_matrix.matrix),
            },
            "quality_masks": {k: _array_to_list(v) for k, v in self.quality_masks.items()},
            "ancillary": self.ancillary,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Sample":
        spectrum_data = data.get("spectrum", {})
        spectrum = Spectrum(
            wavelength_nm=np.asarray(spectrum_data.get("wavelength_nm", []), dtype=float),
            values=np.asarray(spectrum_data.get("values", []), dtype=float),
            kind=spectrum_data.get("kind", "radiance"),
        )
        acquisition_time = data.get("acquisition_time")
        if isinstance(acquisition_time, str):
            try:
                acquisition_time = datetime.fromisoformat(acquisition_time)
            except ValueError:
                pass

        band_meta = data.get("band_meta")
        srf_matrix = data.get("srf_matrix")
        srf_obj = None
        if isinstance(srf_matrix, Mapping):
            srf_obj = SRFMatrix(
                wavelength_nm=np.asarray(srf_matrix.get("wavelength_nm", []), dtype=float),
                matrix=np.asarray(srf_matrix.get("matrix", []), dtype=float),
            )

        quality_masks = {
            str(k): np.asarray(v, dtype=bool) for k, v in (data.get("quality_masks") or {}).items()
        }

        return cls(
            spectrum=spectrum,
            sensor_id=str(data.get("sensor_id", "unknown")),
            acquisition_time=acquisition_time,
            geo=data.get("geo"),
            viewing_geometry=data.get("viewing_geometry"),
            band_meta=band_meta,
            srf_matrix=srf_obj,
            quality_masks=quality_masks,
            ancillary=dict(data.get("ancillary", {})),
        )

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
        spectrum = Spectrum(
            wavelength_nm=np.asarray(wavelength_nm, dtype=float), values=values, kind=kind
        )
        return cls(spectrum=spectrum, sensor_id=sensor_id, **kwargs)

    @staticmethod
    def _ensure_spectrum(spectrum: Any) -> Spectrum:
        if isinstance(spectrum, Spectrum):
            return spectrum

        # Lightweight conversion path for legacy ``alchemi.types.Spectrum`` instances
        wavelength_nm = getattr(spectrum, "wavelength_nm", None)
        values = getattr(spectrum, "values", None)
        kind = getattr(spectrum, "kind", None)
        if wavelength_nm is None or values is None or kind is None:
            raise TypeError("spectrum must be a Spectrum or legacy spectrum-like object")

        kind_value = kind.value if hasattr(kind, "value") else kind
        normalized_kind = str(kind_value)
        if normalized_kind.lower().startswith("brightness"):
            normalized_kind = "BT"

        return Spectrum(
            wavelength_nm=np.asarray(wavelength_nm, dtype=float),
            values=np.asarray(values, dtype=float),
            kind=normalized_kind,
        )

    @staticmethod
    def _normalize_geo(
        geo: GeoMeta | Mapping[str, float] | Tuple[float, float, float] | None,
    ) -> GeoMeta | None:
        if geo is None:
            return None
        if isinstance(geo, GeoMeta):
            return geo
        if isinstance(geo, tuple):
            if len(geo) != 3:
                raise ValueError("geo tuples must contain (lat, lon, elev)")
            lat, lon, elev = geo
            return GeoMeta(
                lat=float(lat), lon=float(lon), elev=None if elev is None else float(elev)
            )
        if isinstance(geo, Mapping):
            geo_dict = dict(geo)
            lat_val = geo_dict.get("lat") if "lat" in geo_dict else geo_dict.get("latitude")
            lon_val = geo_dict.get("lon") if "lon" in geo_dict else geo_dict.get("longitude")
            if lat_val is None or lon_val is None:
                raise ValueError("geo mapping must provide lat/lon keys")
            elev_val = geo_dict.get("elev") if "elev" in geo_dict else geo_dict.get("elevation")
            return GeoMeta(
                lat=float(lat_val),
                lon=float(lon_val),
                elev=None if elev_val is None else float(elev_val),
            )
        raise TypeError("geo must be a GeoMeta, mapping, tuple, or None")
