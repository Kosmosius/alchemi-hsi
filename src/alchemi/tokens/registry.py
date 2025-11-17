"""Tokeniser registry that exposes sensor-aware presets."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, TypedDict, cast

from .band_tokenizer import BandTokConfig, BandTokenizer, ValueNorm

AxisUnit = Literal["nm", "cm-1"]


class _BandTokOverrides(TypedDict, total=False):
    n_fourier_frequencies: int
    value_norm: ValueNorm
    lambda_unit: AxisUnit
    include_width: bool
    include_srf_embed: bool
    token_dim: int
    axis_norm: Literal["zscore", "minmax"]
    srf_embed_dim: int
    projection_seed: int
    sensor_id: str | None
    epsilon: float


_SENSOR_PRESETS: dict[str, _BandTokOverrides] = {
    "emit": {"token_dim": 96, "include_srf_embed": True, "sensor_id": "emit"},
    "enmap": {"token_dim": 96, "include_srf_embed": True, "sensor_id": "enmap"},
    "avirisng": {"token_dim": 96, "include_srf_embed": True, "sensor_id": "avirisng"},
    "hytes": {
        "token_dim": 80,
        "include_srf_embed": True,
        "sensor_id": "hytes",
        "lambda_unit": "cm-1",
    },
}


@lru_cache(maxsize=32)
def get_default_tokenizer(sensor_id: str | None, axis_unit: AxisUnit) -> BandTokenizer:
    """Return a cached tokenizer preset for ``sensor_id`` and ``axis_unit``."""

    axis_unit_lower = axis_unit.lower()
    if axis_unit_lower not in {"nm", "cm-1"}:
        msg = "axis_unit must be 'nm' or 'cm-1'"
        raise ValueError(msg)
    axis_unit_checked = cast(AxisUnit, axis_unit_lower)

    sensor = sensor_id.lower() if sensor_id else None
    base_kwargs: _BandTokOverrides = {
        "lambda_unit": axis_unit_checked,
        "value_norm": "per_spectrum_zscore",
        "include_width": True,
        "token_dim": 72,
    }
    if sensor in _SENSOR_PRESETS:
        base_kwargs.update(_SENSOR_PRESETS[sensor])
    config = BandTokConfig(**base_kwargs)
    return BandTokenizer(config=config)


__all__ = ["get_default_tokenizer"]
