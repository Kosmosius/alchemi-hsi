"""YAML configuration loader with reproducibility hooks.

This module provides a small Hydra-style loader that composes multiple YAML
files and optional in-memory overrides. The packaged ``defaults.yaml`` serves
as a base layer for spectra-based experiments.
"""
from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .seed import seed_everything

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "defaults.yaml"


def _deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping")
    return data


def load_config(
    *paths: str | Path,
    overrides: Mapping[str, Any] | None = None,
    include_defaults: bool = True,
) -> dict[str, Any]:
    """Load one or more YAML files and merge them Hydra-style.

    Later paths take precedence over earlier ones. When ``include_defaults`` is
    ``True`` (the default), :mod:`spectra`'s packaged ``defaults.yaml`` is used
    as the base layer. ``overrides`` may contain in-memory values that will be
    merged last.
    """
    cfg: dict[str, Any] = {}

    if include_defaults and DEFAULT_CONFIG_PATH.exists():
        cfg = _deep_update(cfg, _load_yaml(DEFAULT_CONFIG_PATH))

    for path in paths:
        cfg = _deep_update(cfg, _load_yaml(Path(path)))

    if overrides:
        cfg = _deep_update(cfg, overrides)

    return cfg


def apply_repro_settings(cfg: Mapping[str, Any]) -> None:
    """Apply seed and determinism settings from a loaded config.

    This is a convenience wrapper around :func:`seed_everything`.
    """
    seed = int(cfg.get("seed", 0))
    deterministic = bool(cfg.get("deterministic", False))
    seed_everything(seed=seed, deterministic=deterministic)
    LOGGER.info("Configured reproducibility: seed=%d deterministic=%s", seed, deterministic)
