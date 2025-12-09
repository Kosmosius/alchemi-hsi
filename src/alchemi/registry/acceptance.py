from __future__ import annotations

from typing import Literal

import numpy as np

from .sensors import SensorSpec
from ..types import SRFMatrix


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


def evaluate_sensor_acceptance(
    sensor_spec: SensorSpec, srf_matrix: SRFMatrix
) -> Literal["accepted", "needs_finetune", "out_of_scope"]:
    """Coarse acceptance test for sensor SRFs.

    The thresholds are intentionally generous and should be refined based on
    SRF robustness experiments. TODO: tune coverage and FWHM tolerances using
    downstream reconstruction benchmarks.
    """

    min_nm, max_nm = sensor_spec.wavelength_range_nm
    if srf_matrix.centers_nm.min() > min_nm + 50 or srf_matrix.centers_nm.max() < max_nm - 50:
        return "out_of_scope"

    if srf_matrix.centers_nm.size != sensor_spec.expected_band_count:
        return "needs_finetune"

    centers_diff = np.abs(srf_matrix.centers_nm - sensor_spec.band_centers_nm)
    center_tol = 15.0
    if np.any(centers_diff > center_tol):
        return "needs_finetune"

    estimated_fwhm = np.array(
        [
            _estimate_fwhm(nm, resp)
            for nm, resp in zip(srf_matrix.bands_nm, srf_matrix.bands_resp, strict=True)
        ]
    )
    if estimated_fwhm.shape != sensor_spec.band_widths_nm.shape:
        return "needs_finetune"
    width_diff = np.abs(estimated_fwhm - sensor_spec.band_widths_nm)
    width_tol = 20.0
    if np.any(width_diff > width_tol):
        return "needs_finetune"

    if sensor_spec.absorption_windows_nm:
        for start, end in sensor_spec.absorption_windows_nm:
            if (
                srf_matrix.centers_nm[
                    (srf_matrix.centers_nm >= start) & (srf_matrix.centers_nm <= end)
                ].size
                == 0
            ):
                return "needs_finetune"

    return "accepted"


__all__ = ["evaluate_sensor_acceptance"]
