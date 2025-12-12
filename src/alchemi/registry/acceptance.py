from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np

from .sensors import SensorSpec
from ..types import SRFMatrix

# Acceptance thresholds are intentionally conservative but configurable. They
# capture the Section 6 criteria in a lightweight, testable form.
REFLECTIVE_WINDOWS = (
    {"name": "visible", "range": (400.0, 720.0), "min_bands": 5},
    {"name": "nir", "range": (900.0, 1350.0), "min_bands": 5},
    {"name": "mineral_swir", "range": (1950.0, 2450.0), "min_bands": 8},
    {"name": "methane", "range": (2290.0, 2400.0), "min_bands": 3},
)

LWIR_WINDOWS = (
    {"name": "lwir_core", "range": (8000.0, 12000.0), "min_bands": 20},
)

REFLECTIVE_WIDTH_BOUNDS = (1.5, 70.0)
LWIR_WIDTH_BOUNDS = (10.0, 150.0)

MAX_GAP_WARN_NM = 60.0
MAX_GAP_REJECT_NM = 120.0

CENTER_TOL_WARN_NM = 10.0
CENTER_TOL_REJECT_NM = 20.0

WIDTH_REL_TOL_WARN = 0.35
WIDTH_REL_TOL_REJECT = 0.8

NEGATIVE_LOBE_TOL = -1e-3
MULTILOBE_HALF_MAX = 0.5


class AcceptanceVerdict(Enum):
    ACCEPT = "accept"
    ACCEPT_WITH_WARNINGS = "accept_with_warnings"
    REJECT = "reject"


@dataclass
class AcceptanceReport:
    verdict: AcceptanceVerdict
    warnings: list[str]
    rejections: list[str]


def _estimate_fwhm(nm: np.ndarray, resp: np.ndarray) -> float:
    if nm.size == 0 or resp.size == 0:
        return float("nan")
    peak = float(np.nanmax(resp))
    if not np.isfinite(peak) or peak <= 0.0:
        return float("nan")
    half = 0.5 * peak
    above = np.where(resp >= half)[0]
    if above.size == 0:
        return float("nan")
    return float(nm[above[-1]] - nm[above[0]])


def _count_peaks(resp: np.ndarray) -> int:
    if resp.size == 0:
        return 0
    resp_norm = resp / np.nanmax(resp)
    above = resp_norm >= MULTILOBE_HALF_MAX
    if not np.any(above):
        return 0
    diffs = np.diff(above.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    if above[0]:
        starts = np.concatenate(([-1], starts))
    if above[-1]:
        ends = np.concatenate((ends, [resp.size - 1]))
    return int(min(starts.size, ends.size))


def _band_mask(spec: SensorSpec, srf: SRFMatrix) -> np.ndarray:
    mask = np.ones_like(srf.centers_nm, dtype=bool)
    if spec.bad_band_mask is not None:
        mask &= ~spec.bad_band_mask
    if srf.bad_band_mask is not None:
        mask &= ~srf.bad_band_mask
    return mask


def _max_gap_nm(centers: np.ndarray, widths: np.ndarray) -> float:
    starts = centers - 0.5 * widths
    ends = centers + 0.5 * widths
    order = np.argsort(starts)
    starts = starts[order]
    ends = ends[order]
    if starts.size <= 1:
        return 0.0
    return float(np.max(starts[1:] - ends[:-1]))


def _coverage_checks(
    centers: np.ndarray, windows: Iterable[dict[str, float | int]], spectral_range: tuple[float, float]
) -> list[str]:
    failures: list[str] = []
    min_nm, max_nm = spectral_range
    for window in windows:
        start, end = window["range"]  # type: ignore[index]
        min_bands = int(window["min_bands"])  # type: ignore[index]
        if float(end) < min_nm or float(start) > max_nm:
            continue
        overlap_start = max(float(start), min_nm)
        overlap_end = min(float(end), max_nm)
        overlap_width = overlap_end - overlap_start
        required = max(1, int(np.ceil(min_bands * overlap_width / (float(end) - float(start)))))
        in_window = (centers >= overlap_start) & (centers <= overlap_end)
        if int(np.count_nonzero(in_window)) < required:
            failures.append(
                f"Insufficient coverage of {window['name']} window {overlap_start}-{overlap_end} nm"
            )
    return failures


def evaluate_sensor_acceptance(
    sensor_spec: SensorSpec, srf_matrix: SRFMatrix
) -> AcceptanceReport:
    """Acceptance test for sensor SRFs against Section 6 criteria."""

    warnings: list[str] = []
    rejections: list[str] = []

    mask = _band_mask(sensor_spec, srf_matrix)
    centers = srf_matrix.centers_nm[mask]
    if centers.size == 0:
        rejections.append("No valid bands remain after masking")
        return AcceptanceReport(AcceptanceVerdict.REJECT, warnings, rejections)

    min_nm, max_nm = sensor_spec.wavelength_range_nm
    if centers.min() > min_nm + 50 or centers.max() < max_nm - 50:
        rejections.append("Sensor does not cover documented wavelength range")

    if mask.size != sensor_spec.expected_band_count:
        warnings.append("Band mask length does not match expected band count")

    if srf_matrix.centers_nm.size != sensor_spec.expected_band_count:
        rejections.append("Band count does not match specification")

    centers_diff = np.abs(srf_matrix.centers_nm - sensor_spec.band_centers_nm)
    max_center_diff = float(np.nanmax(centers_diff))
    if max_center_diff >= CENTER_TOL_REJECT_NM:
        rejections.append("Band centers deviate beyond tolerance")
    elif max_center_diff > CENTER_TOL_WARN_NM:
        warnings.append("Band centers differ from spec beyond warning threshold")

    estimated_fwhm = np.array(
        [
            _estimate_fwhm(nm, resp)
            for nm, resp in zip(srf_matrix.bands_nm, srf_matrix.bands_resp, strict=True)
        ]
    )
    if estimated_fwhm.shape != sensor_spec.band_widths_nm.shape:
        rejections.append("Estimated FWHM shape mismatch with spec")
        return AcceptanceReport(_verdict(warnings, rejections), warnings, rejections)

    width_diff = np.abs(estimated_fwhm - sensor_spec.band_widths_nm)
    rel_width_err = width_diff / np.maximum(sensor_spec.band_widths_nm, 1e-6)
    if np.any(rel_width_err > WIDTH_REL_TOL_REJECT):
        rejections.append("Band widths deviate beyond tolerance")
    elif np.any(rel_width_err > WIDTH_REL_TOL_WARN):
        warnings.append("Band widths differ from spec beyond warning threshold")

    domain = "lwir" if sensor_spec.wavelength_range_nm[1] > 5000.0 else "reflective"
    width_bounds = LWIR_WIDTH_BOUNDS if domain == "lwir" else REFLECTIVE_WIDTH_BOUNDS
    too_narrow = estimated_fwhm < width_bounds[0]
    too_wide = estimated_fwhm > width_bounds[1]
    if np.any(too_narrow):
        rejections.append("Bands are unrealistically narrow")
    if np.any(too_wide):
        rejections.append("Bands are excessively broad")

    widths_for_gap = estimated_fwhm if np.all(np.isfinite(estimated_fwhm)) else sensor_spec.band_widths_nm
    gap_nm = _max_gap_nm(centers, widths_for_gap[mask])
    if gap_nm > MAX_GAP_REJECT_NM:
        rejections.append(f"Gaps between bands are too large ({gap_nm:.1f} nm)")
    elif gap_nm > MAX_GAP_WARN_NM:
        warnings.append(f"Large gaps between bands ({gap_nm:.1f} nm)")

    windows = LWIR_WINDOWS if domain == "lwir" else REFLECTIVE_WINDOWS
    rejections.extend(_coverage_checks(centers, windows, sensor_spec.wavelength_range_nm))

    for resp in srf_matrix.bands_resp:
        if np.nanmin(resp) < NEGATIVE_LOBE_TOL:
            rejections.append("SRF contains negative lobes")
            break

    peak_counts = [
        _count_peaks(np.asarray(resp, dtype=np.float64))
        for resp in srf_matrix.bands_resp
    ]
    if any(count > 1 for count in peak_counts):
        rejections.append("SRF shows strong multi-lobed responses")

    if sensor_spec.absorption_windows_nm:
        for start, end in sensor_spec.absorption_windows_nm:
            in_window = (centers >= start) & (centers <= end)
            if not bool(np.any(in_window)):
                warnings.append(f"No bands covering absorption window {start}-{end} nm")

    return AcceptanceReport(_verdict(warnings, rejections), warnings, rejections)


def _verdict(warnings: list[str], rejections: list[str]) -> AcceptanceVerdict:
    if rejections:
        return AcceptanceVerdict.REJECT
    if warnings:
        return AcceptanceVerdict.ACCEPT_WITH_WARNINGS
    return AcceptanceVerdict.ACCEPT


__all__ = ["AcceptanceReport", "AcceptanceVerdict", "evaluate_sensor_acceptance"]
