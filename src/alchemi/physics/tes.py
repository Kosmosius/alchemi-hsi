"""LWIR utilities and TES placeholder implementations."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from alchemi.physics import planck
from alchemi.spectral import Sample
from alchemi.types import (
    QuantityKind,
    ReflectanceUnits,
    Spectrum,
    TemperatureUnits,
    ValueUnits,
)

__all__ = [
    "radiance_spectrum_to_bt_spectrum",
    "bt_spectrum_to_radiance_spectrum",
    "radiance_sample_to_bt_sample",
    "compute_lwir_emissivity_proxy",
    "lwir_pipeline_for_sample",
    "tes_lwirt",
]


def radiance_spectrum_to_bt_spectrum(
    spectrum: Spectrum,
    *,
    srf_matrix: np.ndarray | None = None,
    srf_wavelength_nm: np.ndarray | None = None,
    method: str = "central_lambda",
    temps_grid_K: np.ndarray | None = None,
    strict: bool = False,
) -> Spectrum:
    """Convert an LWIR radiance :class:`Spectrum` to brightness temperature.

    ``method`` mirrors :func:`alchemi.physics.planck.radiance_spectrum_to_bt`:

    - ``"central_lambda"`` (default): uses SRF effective centres or native
      wavelengths. Recommended for narrow bands or when SRFs are unavailable.
    - ``"band"``: performs SRF-weighted band inversion and requires
      ``srf_matrix`` and ``srf_wavelength_nm``. Recommended for HyTES-like or
      other broad SRFs where band-averaged BTs are needed. ``temps_grid_K`` can
      tune the inversion grid and ``strict`` controls out-of-range handling.
    """

    if spectrum.kind != QuantityKind.RADIANCE:
        raise ValueError("Input spectrum must have kind radiance")

    bt = planck.radiance_spectrum_to_bt(
        spectrum,
        srf_matrix=srf_matrix,
        srf_wavelength_nm=srf_wavelength_nm,
        method=method,
        temps_grid_K=temps_grid_K,
        strict=strict,
    )
    return bt


def bt_spectrum_to_radiance_spectrum(
    spectrum_bt: Spectrum,
    *,
    srf_matrix: np.ndarray | None = None,
    srf_wavelength_nm: np.ndarray | None = None,
    method: str = "central_lambda",
) -> Spectrum:
    """Convert a brightness-temperature :class:`Spectrum` back to radiance.

    ``method`` mirrors :func:`alchemi.physics.planck.bt_spectrum_to_radiance`:

    - ``"central_lambda"`` (default): Planck evaluation at SRF effective centres
      or native wavelengths.
    - ``"band"``: SRF-based band-averaged radiance, requiring aligned SRF data.
    """

    if spectrum_bt.kind != QuantityKind.BRIGHTNESS_T:
        raise ValueError("Input spectrum must have kind brightness temperature")

    return planck.bt_spectrum_to_radiance(
        spectrum_bt,
        srf_matrix=srf_matrix,
        srf_wavelength_nm=srf_wavelength_nm,
        method=method,
    )


def radiance_sample_to_bt_sample(
    sample: Sample,
    *,
    srf_matrix: np.ndarray | None = None,
    srf_wavelength_nm: np.ndarray | None = None,
    method: str = "central_lambda",
    temps_grid_K: np.ndarray | None = None,
    strict: bool = False,
) -> Sample:
    """Convert a radiance :class:`Sample` to brightness temperature.

    ``method`` follows :func:`radiance_spectrum_to_bt_spectrum`; ``"band"``
    requires SRF information to compute SRF-averaged BTs.
    """

    matrix = srf_matrix
    wl = srf_wavelength_nm
    if matrix is None and sample.srf_matrix is not None:
        matrix = sample.srf_matrix.matrix
        wl = sample.srf_matrix.wavelength_nm

    bt_spectrum = radiance_spectrum_to_bt_spectrum(
        sample.spectrum,
        srf_matrix=matrix,
        srf_wavelength_nm=wl,
        method=method,
        temps_grid_K=temps_grid_K,
        strict=strict,
    )

    return Sample(
        spectrum=bt_spectrum,
        sensor_id=sample.sensor_id,
        acquisition_time=sample.acquisition_time,
        geo=sample.geo,
        viewing_geometry=sample.viewing_geometry,
        band_meta=sample.band_meta,
        srf_matrix=sample.srf_matrix,
        quality_masks=sample.quality_masks,
        ancillary=sample.ancillary,
    )


def compute_lwir_emissivity_proxy(
    bt_spectrum: Spectrum,
) -> tuple[float | np.ndarray, Spectrum]:
    """Compute LWIR brightness-temperature shape proxies.

    The TES-lite proxy used in v1.1 treats the maximum per-pixel brightness
    temperature as a crude temperature proxy (``T_proxy``) and normalises the
    spectrum to obtain a relative shape proxy ``ε̂_k = T_b,k / T_proxy``. This is
    **not** a physical emissivity estimate; it captures only relative spectral
    shape for qualitative anomaly detection and coarse mapping.
    """

    if bt_spectrum.kind != QuantityKind.BRIGHTNESS_T:
        raise ValueError("Input spectrum must be a brightness temperature spectrum")
    if bt_spectrum.units not in {
        TemperatureUnits.KELVIN,
        TemperatureUnits.KELVIN.value,
        ValueUnits.TEMPERATURE_K,
    }:
        raise ValueError("Brightness temperature spectrum must be in Kelvin")

    values = np.asarray(bt_spectrum.values, dtype=np.float64)
    if bt_spectrum.mask is not None:
        masked = ~np.asarray(bt_spectrum.mask, dtype=bool)
        values = np.where(masked, np.nan, values)

    finite_mask = np.isfinite(values)
    if values.ndim == 1:
        if not np.any(finite_mask):
            raise ValueError("Brightness temperature spectrum contains only NaNs")
        T_proxy = float(np.nanmax(values))
    else:
        valid_pixels = np.any(finite_mask, axis=-1)
        if not np.all(valid_pixels):
            raise ValueError("At least one pixel contains only NaNs")
        T_proxy = np.nanmax(values, axis=-1)

    if np.any(T_proxy <= 0):
        raise ValueError("Proxy temperature must be positive")

    emissivity_values = values / np.expand_dims(T_proxy, axis=-1)
    meta = dict(bt_spectrum.meta)
    meta["role"] = "lwir_emissivity_proxy"

    emissivity_spectrum = Spectrum.from_reflectance(
        bt_spectrum.wavelengths,
        emissivity_values,
        units=ReflectanceUnits.FRACTION,
        mask=bt_spectrum.mask,
        meta=meta,
    )

    return (T_proxy if np.ndim(T_proxy) else float(T_proxy)), emissivity_spectrum


def lwir_pipeline_for_sample(
    sample: Sample,
    *,
    srf_matrix: np.ndarray | None = None,
    srf_wavelength_nm: np.ndarray | None = None,
    method: str = "central_lambda",
) -> Sample:
    """Attach LWIR TES-lite proxies to a sample.

    This pipeline ingests radiance or brightness-temperature spectra, converts
    radiance to per-band BT using :mod:`alchemi.physics.planck`, and derives the
    TES-lite proxies ``T_proxy = max_k T_b,k`` and ``ε̂_k = T_b,k / T_proxy``.
    Both quantities are **proxies only**—they capture spectral shape for anomaly
    detection, coarse land cover cues, and qualitative temperature mapping. v1.1
    explicitly does **not** perform full TES retrieval.
    """

    if sample.spectrum.kind == QuantityKind.RADIANCE:
        bt_sample = radiance_sample_to_bt_sample(
            sample,
            srf_matrix=srf_matrix,
            srf_wavelength_nm=srf_wavelength_nm,
            method=method,
        )
    elif sample.spectrum.kind == QuantityKind.BRIGHTNESS_T:
        bt_sample = sample
    else:
        raise ValueError("Sample spectrum must be radiance or brightness temperature")

    T_proxy, emissivity_proxy = compute_lwir_emissivity_proxy(bt_sample.spectrum)

    ancillary = dict(sample.ancillary)
    ancillary.update(
        {
            "lwir_bt_spectrum": bt_sample.spectrum,
            "lwir_emissivity_proxy": emissivity_proxy,
            "lwir_T_proxy_K": T_proxy,
        }
    )

    return Sample(
        spectrum=sample.spectrum,
        sensor_id=sample.sensor_id,
        acquisition_time=sample.acquisition_time,
        geo=sample.geo,
        viewing_geometry=sample.viewing_geometry,
        band_meta=sample.band_meta,
        srf_matrix=sample.srf_matrix,
        quality_masks=sample.quality_masks,
        ancillary=ancillary,
    )


def tes_lwirt(
    spectrum: Spectrum, ancillary: dict
) -> Tuple[np.ndarray, Spectrum, np.ndarray, np.ndarray]:
    """Perform TES for LWIR radiance spectra.

    Parameters
    ----------
    spectrum:
        Longwave infrared radiance spectrum (typically microns to nm converted).
    ancillary:
        Dictionary of supporting information (e.g. view angles, atmospheric
        profiles). Keys are yet to be finalised.

    Returns
    -------
    Tuple[np.ndarray, Spectrum, np.ndarray, np.ndarray]
        Tuple containing estimated temperature(s), emissivity spectrum,
        uncertainties on temperature, and uncertainties on emissivity.

    Notes
    -----
    This is a placeholder that documents the intended API. A future
    implementation should follow TES literature (e.g., ASTER TES) and provide
    spectral smoothing, NEM/RMSE-based emissivity normalisation, and uncertainty
    propagation.
    """

    raise NotImplementedError("TES retrieval is not yet implemented")
