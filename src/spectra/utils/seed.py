"""Reproducibility utilities for (possibly distributed) experiments.

This module centralises seeding so that Python, NumPy, and PyTorch RNGs are
aligned across distributed processes. It also exposes helpers for configuring
determinism and persisting boolean masks together with the seed used to
generate them.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

_GLOBAL_SEED: int | None = None
_EFFECTIVE_SEED: int | None = None
_RANK: int | None = None
_MASK_RNG: np.random.Generator | None = None


def _detect_rank() -> int:
    """Return the distributed rank if available, otherwise zero.

    When torch.distributed is available but not initialised we fall back to
    common environment variables such as ``LOCAL_RANK`` or ``RANK`` so that
    spawned subprocesses still receive unique seeds.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            return int(torch.distributed.get_rank())
        except (RuntimeError, ValueError):
            # Distributed is available but rank cannot be resolved.
            pass

    for env_key in ("LOCAL_RANK", "RANK"):
        raw = os.environ.get(env_key)
        if raw is not None:
            try:
                return int(raw)
            except ValueError:
                continue
    return 0


def configure_determinism(enabled: bool) -> None:
    """Toggle deterministic kernels for PyTorch backends.

    When enabled cuDNN benchmarking is disabled to avoid nondeterministic kernel
    selection, and deterministic algorithms are requested when available.
    """
    torch.backends.cudnn.deterministic = bool(enabled)
    torch.backends.cudnn.benchmark = not bool(enabled)
    try:
        torch.use_deterministic_algorithms(enabled, warn_only=True)
    except (AttributeError, TypeError):
        if enabled:
            LOGGER.warning(
                "Deterministic algorithms requested but not fully supported by this PyTorch build.",
            )


def seed_everything(
    seed: int,
    deterministic: bool = False,
    *,
    logger: logging.Logger | None = None,
) -> np.random.Generator:
    """Seed Python, NumPy, and PyTorch RNGs in a DDP-safe manner.

    Parameters
    ----------
    seed:
        The base seed to use for all RNGs.
    deterministic:
        When ``True``, configure cuDNN for deterministic behavior and disable
        benchmarking. This may reduce performance but improves reproducibility.
    logger:
        Optional logger used to record the seed configuration.

    Returns
    -------
    numpy.random.Generator
        A NumPy generator initialized with the (possibly rank-adjusted) seed.
    """
    global _GLOBAL_SEED, _EFFECTIVE_SEED, _RANK, _MASK_RNG

    rank = _detect_rank()
    full_seed = int(seed) + int(rank)

    _GLOBAL_SEED = int(seed)
    _EFFECTIVE_SEED = full_seed
    _RANK = rank

    log = logger or LOGGER
    log.info(
        "Seeding RNGs",
        extra={
            "seed": _GLOBAL_SEED,
            "rank": _RANK,
            "full_seed": _EFFECTIVE_SEED,
            "deterministic": bool(deterministic),
        },
    )

    random.seed(full_seed)
    generator = np.random.default_rng(full_seed)
    _MASK_RNG = generator

    torch.manual_seed(full_seed)
    torch.cuda.manual_seed_all(full_seed)
    os.environ["PYTHONHASHSEED"] = str(full_seed)

    configure_determinism(deterministic)

    return generator


def seed_worker(worker_id: int) -> None:
    """Initialize a DataLoader worker with a distinct, rank-adjusted seed.

    This mirrors PyTorch's recommended approach for deterministic
    ``DataLoader`` workers when a base seed has already been set via
    :func:`seed_everything`.
    """
    worker_seed = (torch.initial_seed() + worker_id) % np.iinfo(np.int32).max
    random.seed(worker_seed)


def mask_rng() -> np.random.Generator:
    """Return the NumPy generator reserved for mask sampling.

    ``seed_everything`` must be called before this helper; otherwise a
    ``RuntimeError`` is raised to prevent silent nondeterminism.
    """
    if _MASK_RNG is None:
        raise RuntimeError("seed_everything must be called before mask_rng()")
    return _MASK_RNG


def persist_mask(mask: np.ndarray | list[int] | list[bool], path: str | Path) -> None:
    """Persist a boolean mask to JSON alongside the seeds used to create it."""
    if _GLOBAL_SEED is None or _EFFECTIVE_SEED is None:
        raise RuntimeError("seed_everything must be called before persist_mask()")

    array = np.asarray(mask, dtype=bool)
    payload: dict[str, Any] = {
        "seed": _GLOBAL_SEED,
        "rank": _RANK,
        "effective_seed": _EFFECTIVE_SEED,
        "total": int(array.size),
        "mask": array.tolist(),
    }

    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(payload, indent=2))
    LOGGER.info("Saved mask with seed=%d to %s", _GLOBAL_SEED, dst)
