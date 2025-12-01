"""Simple radiative-transfer regime tagging."""

from __future__ import annotations

from typing import Literal

__all__ = ["classify_rt_regime"]

RegimeLabel = Literal["trusted_swir", "heavy_atmosphere"]


def classify_rt_regime(
    water_vapour_proxy: float,
    aerosol_proxy: float,
    solar_zenith_deg: float,
    view_zenith_deg: float,
    relative_azimuth_deg: float | None = None,
) -> RegimeLabel:
    """Classify an observation into a coarse radiative-transfer regime.

    This intentionally conservative heuristic separates cases where SWIR-based
    retrievals are expected to be trustworthy from scenes dominated by heavy
    atmospheric loading. Thresholds are provisional and should be refined once
    more validation data become available.

    Parameters
    ----------
    water_vapour_proxy:
        Proxy for precipitable water vapour (e.g. g/cm²). Values above ~3 are
        treated as heavy.
    aerosol_proxy:
        Proxy for aerosol optical depth. Values above ~0.25 are treated as
        heavy aerosol.
    solar_zenith_deg, view_zenith_deg:
        Angles in degrees. Very oblique geometries are downgraded to heavy.
    relative_azimuth_deg:
        Optional sun–sensor relative azimuth; currently unused but retained for
        future refinements.
    """

    heavy_water = water_vapour_proxy > 3.0  # TODO: calibrate thresholds
    heavy_aerosol = aerosol_proxy > 0.25  # TODO: calibrate thresholds
    oblique_geom = (solar_zenith_deg > 70.0) or (view_zenith_deg > 50.0)

    if heavy_water or heavy_aerosol or oblique_geom:
        return "heavy_atmosphere"

    return "trusted_swir"
