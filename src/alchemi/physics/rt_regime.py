"""Simple SWIR radiative-transfer regime tagging.

Implements the ``trusted`` vs ``heavy`` atmosphere heuristic from Section 5.3.
The classifier is intentionally conservative and uses common geometry and
atmospheric proxies (solar/view zenith, PWV, AOD, cloud/haze flags) to decide
when simplified TOA reflectance formulas are applicable. Entry points:

* :func:`classify_swir_regime` for scalar or vectorised metadata inputs.
* :func:`swir_regime_for_sample` / :func:`swir_regime_for_scene` to pull fields
  from :class:`~alchemi.spectral.sample.Sample` objects or scene dictionaries.
* :func:`attach_swir_regime` / :func:`trusted_swir` helpers for tagging samples.

Outputs are coarse tags (``SWIRRegime.TRUSTED`` or ``SWIRRegime.HEAVY``); they
do not replace full radiative-transfer modelling.
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
from typing import Any, Mapping

import numpy as np

__all__ = [
    "SWIRRegime",
    "classify_swir_regime",
    "swir_regime_for_sample",
    "swir_regime_for_scene",
    "attach_swir_regime",
    "trusted_swir",
    "classify_rt_regime",
    "is_trusted_swir",
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
) -> SWIRRegime | np.ndarray:
    """Classify geometry/atmosphere metadata into trusted vs heavy SWIR regime.

    Scalar inputs return a :class:`SWIRRegime` whereas array-like inputs are
    vectorised and return an ``np.ndarray`` of ``SWIRRegime`` instances. Missing
    metadata (``None``) are treated as "unknown" and therefore do not
    contribute to a heavy classification unless ``has_heavy_cloud_or_haze`` is
    explicitly flagged.
    """

    def _maybe_broadcast(value: Any, fill_value: float) -> tuple[np.ndarray, tuple[int, ...]]:
        if value is None:
            arr = np.asarray(fill_value)
        else:
            arr = np.asarray(value)
        return arr, arr.shape

    candidates = []
    shapes = []
    solar_arr, shape = _maybe_broadcast(solar_zenith_deg, -np.inf)
    candidates.append((solar_arr.astype(float), solar_zenith_max))
    shapes.append(shape)
    view_arr, shape = _maybe_broadcast(view_zenith_deg, -np.inf)
    candidates.append((view_arr.astype(float), view_zenith_max))
    shapes.append(shape)
    pwv_arr, shape = _maybe_broadcast(pwv_cm, -np.inf)
    candidates.append((pwv_arr.astype(float), pwv_max_cm))
    shapes.append(shape)
    aod_arr, shape = _maybe_broadcast(aod_550, -np.inf)
    candidates.append((aod_arr.astype(float), aod550_max))
    shapes.append(shape)

    cloud_arr = np.asarray(False if has_heavy_cloud_or_haze is None else has_heavy_cloud_or_haze)
    shapes.append(cloud_arr.shape)

    shape = ()
    for candidate_shape in shapes:
        if candidate_shape != ():
            shape = np.broadcast_shapes(shape, candidate_shape)

    heavy = np.zeros(shape, dtype=bool)
    for arr, threshold in candidates:
        arr = np.broadcast_to(arr, shape)
        heavy |= arr > threshold

    cloud_flag = np.broadcast_to(cloud_arr, shape)
    heavy |= cloud_flag

    if heavy.shape == ():
        return SWIRRegime.HEAVY if bool(heavy) else SWIRRegime.TRUSTED

    trusted = np.empty(shape, dtype=object)
    trusted[...] = SWIRRegime.TRUSTED
    trusted[heavy] = SWIRRegime.HEAVY
    return trusted


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


def trusted_swir(sample_or_meta: Mapping[str, Any] | Any) -> bool | np.ndarray:
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

    result = swir_regime_for_sample(sample_or_meta)
    if isinstance(result, np.ndarray):
        return result == SWIRRegime.TRUSTED

    return result == SWIRRegime.TRUSTED


def is_trusted_swir(sample_or_meta: Mapping[str, Any] | Any) -> bool | np.ndarray:
    """Alias for :func:`trusted_swir` for readability in diagnostics."""

    return trusted_swir(sample_or_meta)


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
