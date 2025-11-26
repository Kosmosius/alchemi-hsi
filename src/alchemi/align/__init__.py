"""Alignment utilities and losses with lazy attribute loading."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "CycleAlignment",
    "CycleConfig",
    "CycleReconstructionHeads",
    "HyTESNoiseConfig",
    "InfoNCELossOut",
    "LossOut",
    "NoiseConfig",
    "PairBatch",
    "RandomSensorProject",
    "SyntheticAlignmentDataset",
    "build_avirisng_pairs",
    "build_emit_pairs",
    "build_emits_pairs",
    "build_enmap_pairs",
    "build_hytes_pairs",
    "info_nce_symmetric",
]

_EXPORTS = {
    "CycleAlignment": ("alchemi.align.cycle", "CycleAlignment"),
    "CycleConfig": ("alchemi.align.cycle", "CycleConfig"),
    "CycleReconstructionHeads": ("alchemi.align.cycle", "CycleReconstructionHeads"),
    "HyTESNoiseConfig": ("alchemi.align.hytes", "HyTESNoiseConfig"),
    "build_hytes_pairs": ("alchemi.align.hytes", "build_hytes_pairs"),
    "build_emit_pairs": ("alchemi.align.batch_builders", "build_emit_pairs"),
    "build_enmap_pairs": ("alchemi.align.batch_builders", "build_enmap_pairs"),
    "build_avirisng_pairs": ("alchemi.align.batch_builders", "build_avirisng_pairs"),
    "NoiseConfig": ("alchemi.align.batch_builders", "NoiseConfig"),
    "PairBatch": ("alchemi.align.batch_builders", "PairBatch"),
    "build_emits_pairs": ("alchemi.align.batch_builders", "build_emits_pairs"),
    "RandomSensorProject": ("alchemi.align.transforms", "RandomSensorProject"),
    "SyntheticAlignmentDataset": ("alchemi.align.testing", "SyntheticAlignmentDataset"),
    "LossOut": ("alchemi.align.losses", "LossOut"),
    "InfoNCELossOut": ("alchemi.align.losses", "InfoNCELossOut"),
    "info_nce_symmetric": ("alchemi.align.losses", "info_nce_symmetric"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = importlib.import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value
