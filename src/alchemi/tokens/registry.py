"""Lightweight registry exposing tokeniser presets."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from .band_tokenizer import BandTokConfig, BandTokenizer

AxisUnit = Literal["nm", "cm-1"]


@lru_cache(maxsize=8)
def get_default(axis_unit: AxisUnit) -> BandTokenizer:
    """Return a cached :class:`BandTokenizer` preset for the given axis unit."""

    axis_unit = axis_unit.lower()  # normalise aliases
    if axis_unit not in {"nm", "cm-1"}:
        msg = "axis_unit must be either 'nm' or 'cm-1'"
        raise ValueError(msg)

    if axis_unit == "nm":
        config = BandTokConfig(
            value_norm="robust",
            lambda_norm="zscore",
            n_frequencies=8,
            frequency_base=1.0,
        )
    else:
        config = BandTokConfig(
            value_norm="robust",
            lambda_norm="zscore",
            n_frequencies=8,
            frequency_base=0.5,
        )
    return BandTokenizer(config=config)


__all__ = ["get_default"]

