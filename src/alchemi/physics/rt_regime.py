"""Simple SWIR radiative-transfer regime tagging.

This module implements the ``trusted`` vs ``heavy`` atmosphere heuristic from
ALCHEMI Section 5.3. The classifier is intentionally conservative and relies on
common geometry and atmospheric proxies that are typically available in
ancillary metadata.
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
from typing import Any, Mapping

__all__ = [
    "SWIRRegime",
    "classify_swir_regime",
    "swir_regime_for_sample",
    "swir_regime_for_scene",
    "attach_swir_regime",
    "trusted_swir",
    "classify_rt_regime",
]


class SWIRRegime(str, Enum):
    """Coarse SWIR radiative-transfer regimes."""

    TRUSTED = "trusted"
    HEAVY = "heavy"


DEFAULT_SOLAR_ZENITH_MAX = 60.0
DEFAULT_VIEW_ZENITH_MAX = 30.0
DEFAULT_PWV_MAX_CM = 4.0
DEFAULT_AOD550_MAX = 0.35


def classify_swir_regime(
    *,
    solar_zenith_deg: float | None = None,
    view_zenith_deg: float | None = None,
    pwv_cm: float | None = None,
    aod_550: float | None = None,
    has_heavy_cloud_or_haze: bool | None = None,
    solar_zenith_max: float = DEFAULT_SOLAR_ZENITH_MAX,
    view_zenith_max: float = DEFAULT_VIEW_ZENITH_MAX,
    pwv_max_cm: float = DEFAULT_PWV_MAX_CM,
    aod550_max: float = DEFAULT_AOD550_MAX,
) -> SWIRRegime:
    """Classify geometry/atmosphere metadata into trusted vs heavy SWIR regime."""

    if solar_zenith_deg is not None and solar_zenith_deg > solar_zenith_max:
        return SWIRRegime.HEAVY

    if view_zenith_deg is not None and view_zenith_deg > view_zenith_max:
        return SWIRRegime.HEAVY

    if pwv_cm is not None and pwv_cm > pwv_max_cm:
        return SWIRRegime.HEAVY

    if aod_550 is not None and aod_550 > aod550_max:
        return SWIRRegime.HEAVY

    if has_heavy_cloud_or_haze:
        return SWIRRegime.HEAVY

    return SWIRRegime.TRUSTED


def _extract_from_mapping(mapping: Mapping[str, Any] | None, key: str) -> Any:
    if mapping is None:
        return None
    return mapping.get(key)


def swir_regime_for_scene(
    scene_meta: Mapping[str, Any],
    **kwargs: Any,
) -> SWIRRegime:
    """Classify SWIR regime using a scene-level metadata mapping."""

    return classify_swir_regime(
        solar_zenith_deg=_extract_from_mapping(scene_meta, "solar_zenith_deg"),
        view_zenith_deg=_extract_from_mapping(scene_meta, "view_zenith_deg"),
        pwv_cm=_extract_from_mapping(scene_meta, "pwv_cm"),
        aod_550=_extract_from_mapping(scene_meta, "aod_550"),
        has_heavy_cloud_or_haze=_extract_from_mapping(scene_meta, "has_heavy_cloud_or_haze"),
        **kwargs,
    )


def swir_regime_for_sample(sample: Any, **kwargs: Any) -> SWIRRegime:
    """Classify a :class:`~alchemi.spectral.sample.Sample` based on metadata."""

    ancillary = getattr(sample, "ancillary", None) if sample is not None else None
    meta = getattr(sample, "meta", None) if sample is not None else None

    solar_zenith = None
    view_zenith = None
    if getattr(sample, "viewing_geometry", None) is not None:
        solar_zenith = getattr(sample.viewing_geometry, "solar_zenith_deg", None)
        view_zenith = getattr(sample.viewing_geometry, "view_zenith_deg", None)

    if solar_zenith is None:
        solar_zenith = _extract_from_mapping(ancillary, "solar_zenith_deg")
    if view_zenith is None:
        view_zenith = _extract_from_mapping(ancillary, "view_zenith_deg")

    pwv_cm = _extract_from_mapping(ancillary, "pwv_cm")
    if pwv_cm is None:
        pwv_cm = _extract_from_mapping(meta, "pwv_cm")

    aod_550 = _extract_from_mapping(ancillary, "aod_550")
    if aod_550 is None:
        aod_550 = _extract_from_mapping(meta, "aod_550")

    has_heavy_cloud_or_haze = _extract_from_mapping(ancillary, "has_heavy_cloud_or_haze")
    if has_heavy_cloud_or_haze is None:
        has_heavy_cloud_or_haze = _extract_from_mapping(meta, "has_heavy_cloud_or_haze")

    return classify_swir_regime(
        solar_zenith_deg=solar_zenith,
        view_zenith_deg=view_zenith,
        pwv_cm=pwv_cm,
        aod_550=aod_550,
        has_heavy_cloud_or_haze=has_heavy_cloud_or_haze,
        **kwargs,
    )


def attach_swir_regime(sample: Any, **kwargs: Any) -> Any:
    """Return a shallow-copied Sample with ``swir_regime`` stored in ancillary."""

    regime = swir_regime_for_sample(sample, **kwargs)
    ancillary = dict(getattr(sample, "ancillary", {}) or {})
    ancillary["swir_regime"] = regime.value

    try:
        return replace(sample, ancillary=ancillary)
    except Exception:  # pragma: no cover - fallback for non-dataclass-like inputs
        sample_copy = sample
        if hasattr(sample, "__dict__"):
            sample_copy = sample.__class__(**{**sample.__dict__, "ancillary": ancillary})
        return sample_copy


def trusted_swir(sample_or_meta: Mapping[str, Any] | Any) -> bool:
    """Return ``True`` only if the SWIR regime is trusted for the input."""

    if isinstance(sample_or_meta, Mapping):
        stored = sample_or_meta.get("swir_regime")
        if stored is not None:
            return stored == SWIRRegime.TRUSTED.value or stored == SWIRRegime.TRUSTED
        return swir_regime_for_scene(sample_or_meta) == SWIRRegime.TRUSTED

    ancillary = getattr(sample_or_meta, "ancillary", None)
    stored = None if ancillary is None else ancillary.get("swir_regime")
    if stored is not None:
        return stored == SWIRRegime.TRUSTED.value or stored == SWIRRegime.TRUSTED

    return swir_regime_for_sample(sample_or_meta) == SWIRRegime.TRUSTED


def classify_rt_regime(
    water_vapour_proxy: float,
    aerosol_proxy: float,
    solar_zenith_deg: float,
    view_zenith_deg: float,
    relative_azimuth_deg: float | None = None,
) -> SWIRRegime:
    """Backwards-compatible wrapper around :func:`classify_swir_regime`."""

    _ = relative_azimuth_deg  # retained for API compatibility
    return classify_swir_regime(
        solar_zenith_deg=solar_zenith_deg,
        view_zenith_deg=view_zenith_deg,
        pwv_cm=water_vapour_proxy,
        aod_550=aerosol_proxy,
    )
