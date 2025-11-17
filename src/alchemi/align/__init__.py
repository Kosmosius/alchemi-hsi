"""Alignment utilities for constructing training batches, pairing
laboratory spectra with sensor measurements (including HyTES),
and defining alignment and cycle-consistency losses."""

from .cycle import CycleAlignment, CycleConfig, CycleReconstructionHeads
from .losses import LossOut, info_nce_symmetric
from .hytes import HyTESNoiseConfig, build_hytes_pairs
from .batch import build_emit_pairs, build_enmap_pairs
from .batch_builders import NoiseConfig, Pair, build_emits_pairs
from .transforms import RandomSensorProject

__all__ = [
    # Cycle alignment / reconstruction
    "CycleAlignment",
    "CycleConfig",
    "CycleReconstructionHeads",
    # HyTES-specific pairing utilities
    "HyTESNoiseConfig",
    "build_hytes_pairs",
    # General batch builders
    "build_emit_pairs",
    "build_enmap_pairs",
    "NoiseConfig",
    "Pair",
    "build_emits_pairs",
    "RandomSensorProject",
    # Losses
    "LossOut",
    "info_nce_symmetric",
]
