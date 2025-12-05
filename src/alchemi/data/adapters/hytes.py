"""Adapters for HyTES brightness-temperature and radiance products."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr

from alchemi.physics import planck
from alchemi.registry import srfs
from alchemi.spectral import BandMetadata, Sample, Spectrum
from alchemi.spectral.srf import SRFMatrix as DenseSRFMatrix
from alchemi.types import SRFMatrix as LegacySRFMatrix, ValueUnits
from alchemi.srf.utils import build_gaussian_srf_matrix, default_band_widths, validate_srf_alignment

from ..io.hytes import _ensure_kelvin, load_hytes_l1b_bt

__all__ = [
    "iter_hytes_pixels",
    "load_hytes_scene",
    "iter_hytes_radiance_pixels",
    "load_hytes_radiance_scene",
]


_EDGE_BAND_COUNT = 2


def _load_ds(path: str) -> xr.Dataset:
    return load_hytes_l1b_bt(path)


def _coerce_srf_matrix(
    wavelengths_nm: np.ndarray, *, srf_blind: bool
) -> tuple[DenseSRFMatrix | None, str, np.ndarray | None, list[tuple[float, float]] | None, np.ndarray]:
    widths = default_band_widths("hytes", wavelengths_nm)
    if not srf_blind:
        try:
            raw: LegacySRFMatrix | None = srfs.get_srf("hytes")
        except Exception:
            raw = None

        if raw is not None:
            centers = np.asarray(raw.centers_nm, dtype=np.float64)
            if centers.shape[0] == wavelengths_nm.shape[0]:
                matrix = np.zeros((centers.shape[0], wavelengths_nm.shape[0]), dtype=np.float64)
                areas: list[float] = []
                for idx, (nm, resp) in enumerate(zip(raw.bands_nm, raw.bands_resp, strict=True)):
                    nm_arr = np.asarray(nm, dtype=np.float64)
                    resp_arr = np.asarray(resp, dtype=np.float64)
                    matrix[idx, :] = np.interp(wavelengths_nm, nm_arr, resp_arr, left=0.0, right=0.0)
                    area = float(np.trapz(matrix[idx, :], x=wavelengths_nm))
                    areas.append(area)
                    if area > 0:
                        matrix[idx, :] /= area

                if areas and not np.any(np.asarray(areas) <= 0.0):
                    dense = DenseSRFMatrix(wavelength_nm=wavelengths_nm, matrix=matrix)
                    validate_srf_alignment(wavelengths_nm, dense.matrix, centers_nm=wavelengths_nm)
                    provenance = "official"
                    version = str(getattr(raw, "version", ""))
                    if "gaussian" in version.lower():
                        provenance = "gaussian"
                    return dense, provenance, raw.bad_band_mask, raw.bad_band_windows_nm, widths

    dense = build_gaussian_srf_matrix(wavelengths_nm, widths, sensor="hytes")
    return dense, "gaussian", None, None, widths


def _edge_mask(length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    if length == 0:
        return mask
    span = min(_EDGE_BAND_COUNT, max(1, length))
    mask[:span] = True
    mask[-span:] = True
    return mask


def _valid_band_mask(
    wavelengths: np.ndarray,
    *,
    dataset_mask: np.ndarray | None,
    srf_mask: np.ndarray | None,
    srf_windows: list[tuple[float, float]] | None,
    detector_mask: np.ndarray | None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    valid = np.ones_like(wavelengths, dtype=bool)
    quality: Dict[str, np.ndarray] = {}

    if dataset_mask is not None:
        dataset_mask = np.asarray(dataset_mask, dtype=bool)
        valid &= dataset_mask
        quality["band_mask"] = dataset_mask

    if srf_mask is not None:
        srf_bad = np.asarray(srf_mask, dtype=bool)
        valid &= ~srf_bad
        quality["srf_bad_band"] = srf_bad

    edge_mask = _edge_mask(wavelengths.size)
    if np.any(edge_mask):
        valid &= ~edge_mask
        quality["edge_band"] = edge_mask

    if srf_windows is not None:
        window_mask = np.zeros_like(valid, dtype=bool)
        for start, end in srf_windows:
            window_mask |= (wavelengths >= start) & (wavelengths <= end)
        valid &= ~window_mask
        quality["srf_bad_window"] = window_mask

    if detector_mask is not None:
        detector_mask = np.asarray(detector_mask, dtype=bool)
        valid &= ~detector_mask
        quality["bad_detector"] = detector_mask

    quality["valid_band"] = valid
    return valid, quality


def _extract_detector_mask(ds: xr.Dataset) -> np.ndarray | None:
    for key in ("detector_mask", "bad_detector_mask", "bad_detector"):
        if key in ds:
            return np.asarray(ds[key].values, dtype=bool)
    if isinstance(ds.attrs.get("bad_detector_mask"), (list, np.ndarray)):
        return np.asarray(ds.attrs["bad_detector_mask"], dtype=bool)
    return None


def _band_metadata(wavelengths: np.ndarray, valid: np.ndarray, *, srf_source: str) -> BandMetadata:
    return BandMetadata(
        center_nm=np.asarray(wavelengths, dtype=np.float64),
        width_nm=np.full_like(wavelengths, np.nan, dtype=np.float64),
        valid_mask=np.asarray(valid, dtype=bool),
        srf_source=np.full_like(wavelengths, srf_source, dtype=object),
    )


def iter_hytes_pixels(path: str, *, srf_blind: bool = False) -> Iterable[Sample]:
    """Yield HyTES brightness-temperature pixels as :class:`Sample` objects."""

    ds = _load_ds(path)
    bt = ds["brightness_temp"] if "brightness_temp" in ds else ds["bt"]
    wavelengths = np.asarray(ds["wavelength_nm"].values, dtype=np.float64)
    values = np.asarray(bt.values, dtype=np.float64)
    units = bt.attrs.get("units") or ds.attrs.get("brightness_temp_units")
    if isinstance(units, str):
        units = units.strip().lower()
    values = _ensure_kelvin(values, units)

    band_mask = np.asarray(ds["band_mask"].values, dtype=bool) if "band_mask" in ds else None
    detector_mask = _extract_detector_mask(ds)

    srf_matrix, srf_source, srf_bad_mask, srf_windows, widths = _coerce_srf_matrix(
        wavelengths, srf_blind=srf_blind
    )
    valid, quality_base = _valid_band_mask(
        wavelengths,
        dataset_mask=band_mask,
        srf_mask=srf_bad_mask,
        srf_windows=srf_windows,
        detector_mask=detector_mask,
    )
    band_meta = BandMetadata(
        center_nm=np.asarray(wavelengths, dtype=np.float64),
        width_nm=widths,
        valid_mask=np.asarray(valid, dtype=bool),
        srf_source=np.full_like(wavelengths, srf_source, dtype=object),
    )

    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            spec = Spectrum(
                wavelength_nm=wavelengths,
                values=values[y, x, :],
                kind="BT",
                units=ValueUnits.TEMPERATURE_K,
            )
            quality_masks = {name: mask.copy() for name, mask in quality_base.items()}
            sample = Sample(
                spectrum=spec,
                sensor_id="hytes",
                band_meta=band_meta,
                srf_matrix=srf_matrix,
                quality_masks=quality_masks,
                ancillary={
                    "source_path": path,
                    "y": int(y),
                    "x": int(x),
                    "srf_source": srf_source,
                    "srf_mode": "srf-blind" if srf_blind else "srf-aware",
                },
            )
            yield sample


def iter_hytes_radiance_pixels(path: str, *, srf_blind: bool = False) -> Iterable[Sample]:
    """Yield HyTES pixels converted to radiance via Planck inversion."""

    for sample in iter_hytes_pixels(path, srf_blind=srf_blind):
        rad_sample = planck.bt_sample_to_radiance_sample(sample)
        rad_sample.ancillary.setdefault("source_path", sample.ancillary.get("source_path"))
        rad_sample.ancillary.setdefault("y", sample.ancillary.get("y"))
        rad_sample.ancillary.setdefault("x", sample.ancillary.get("x"))
        rad_sample.ancillary.setdefault("srf_source", sample.ancillary.get("srf_source"))
        rad_sample.ancillary.setdefault("srf_mode", sample.ancillary.get("srf_mode"))
        yield rad_sample


def load_hytes_scene(path: str, *, srf_blind: bool = False) -> List[Sample]:
    """Materialise brightness-temperature pixels into memory."""

    return list(iter_hytes_pixels(path, srf_blind=srf_blind))


def load_hytes_radiance_scene(path: str, *, srf_blind: bool = False) -> List[Sample]:
    """Materialise radiance pixels into memory."""

    return list(iter_hytes_radiance_pixels(path, srf_blind=srf_blind))
