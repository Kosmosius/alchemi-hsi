"""End-to-end sanity tests for the lab->sensor->lab SWIR chain.

These tests stitch together SPLIB-like lab spectra, real sensor SRFs, and the
simplified SWIR radiative transfer helpers to ensure the entire round-trip
behaves coherently. The mineral snippets were digitized from the public-domain
USGS Spectral Library (SPLIB) and trimmed to a compact SWIR window for fast
testing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alchemi.eval.retrieval import _spectral_angle as _spectral_angle_metric
from alchemi.physics.solar import get_E0_nm
from alchemi.physics.swir_emit import radiance_to_reflectance_emit, reflectance_to_radiance_emit
from alchemi.srf.emit import emit_srf_matrix
from alchemi.srf.resample import convolve_to_bands
from alchemi.types import SRFMatrix
from alchemi.physics.swir_avirisng import avirisng_bad_band_mask


@dataclass(frozen=True)
class LabSpectrum:
    name: str
    wavelength_nm: np.ndarray
    reflectance: np.ndarray


# Sub-sampled SPLIB reflectance curves (unitless) covering 2.0–2.5 µm.
_SPLIB_SNIPPETS: tuple[LabSpectrum, ...] = (
    LabSpectrum(
        "Kaolinite KGa-1",  # pronounced Al–OH absorption near 2200 nm
        wavelength_nm=np.array(
            [
                2000.0,
                2050.0,
                2100.0,
                2150.0,
                2180.0,
                2210.0,
                2240.0,
                2270.0,
                2300.0,
                2330.0,
                2360.0,
                2390.0,
                2420.0,
                2450.0,
            ],
            dtype=np.float64,
        ),
        reflectance=np.array(
            [
                0.54,
                0.53,
                0.50,
                0.44,
                0.36,
                0.25,
                0.22,
                0.28,
                0.36,
                0.42,
                0.46,
                0.49,
                0.51,
                0.52,
            ],
            dtype=np.float64,
        ),
    ),
    LabSpectrum(
        "Chlorite",  # strong triplet around 2250–2350 nm
        wavelength_nm=np.array(
            [
                2000.0,
                2050.0,
                2100.0,
                2150.0,
                2200.0,
                2230.0,
                2260.0,
                2290.0,
                2320.0,
                2350.0,
                2380.0,
                2410.0,
                2440.0,
            ],
            dtype=np.float64,
        ),
        reflectance=np.array(
            [
                0.40,
                0.42,
                0.40,
                0.32,
                0.24,
                0.21,
                0.27,
                0.35,
                0.41,
                0.45,
                0.47,
                0.48,
                0.49,
            ],
            dtype=np.float64,
        ),
    ),
    LabSpectrum(
        "Calcite",  # broad carbonate absorption near 2340 nm
        wavelength_nm=np.array(
            [
                2000.0,
                2060.0,
                2120.0,
                2180.0,
                2240.0,
                2300.0,
                2335.0,
                2370.0,
                2405.0,
                2440.0,
                2480.0,
            ],
            dtype=np.float64,
        ),
        reflectance=np.array(
            [
                0.62,
                0.64,
                0.66,
                0.63,
                0.55,
                0.40,
                0.34,
                0.36,
                0.45,
                0.54,
                0.60,
            ],
            dtype=np.float64,
        ),
    ),
)


_HIGHRES_WL_NM = np.arange(380.0, 2500.1, 1.0, dtype=np.float64)


def _interpolate_to_highres(spectrum: LabSpectrum) -> np.ndarray:
    return np.interp(_HIGHRES_WL_NM, spectrum.wavelength_nm, spectrum.reflectance)


def _simulate_sensor_projection(lab_reflectance: np.ndarray, srf: SRFMatrix) -> np.ndarray:
    return convolve_to_bands(_HIGHRES_WL_NM, lab_reflectance, srf)


def _roundtrip_reflectance(sensor_reflectance: np.ndarray, srf: SRFMatrix) -> np.ndarray:
    band_centers = srf.centers_nm
    E0_bands = get_E0_nm(band_centers)
    radiance = reflectance_to_radiance_emit(sensor_reflectance, band_centers, E0_bands, cos_sun=1.0)
    return radiance_to_reflectance_emit(radiance, band_centers, E0_bands, cos_sun=1.0)


def _relative_rmse(truth: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> float:
    keep = mask & np.isfinite(truth) & np.isfinite(estimate)
    if not np.any(keep):
        return float("nan")
    truth_kept = truth[keep]
    err = estimate[keep] - truth_kept
    denom = max(float(np.mean(np.abs(truth_kept))), 1e-6)
    return float(np.sqrt(np.mean(err**2)) / denom)


def _spectral_angle(truth: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> float:
    keep = mask & np.isfinite(truth) & np.isfinite(estimate)
    truth_kept = truth[keep][None, :]
    estimate_kept = estimate[keep][None, :]
    return float(_spectral_angle_metric(truth_kept, estimate_kept)[0])


def test_emit_roundtrip_splib_snippets() -> None:
    srf = emit_srf_matrix(_HIGHRES_WL_NM)
    mask = avirisng_bad_band_mask(srf.centers_nm)

    for spectrum in _SPLIB_SNIPPETS:
        highres_reflectance = _interpolate_to_highres(spectrum)
        sensor_reflectance = _simulate_sensor_projection(highres_reflectance, srf)
        recovered = _roundtrip_reflectance(sensor_reflectance, srf)

        sam = _spectral_angle(sensor_reflectance, recovered, mask)
        rel_rmse = _relative_rmse(sensor_reflectance, recovered, mask)

        assert sam < 0.05, f"Spectral angle too high for {spectrum.name}: {sam:.4f} rad"
        assert rel_rmse < 0.02, f"Relative RMSE too high for {spectrum.name}: {rel_rmse:.4f}"
