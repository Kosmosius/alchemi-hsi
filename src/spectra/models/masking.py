from __future__ import annotations

from pathlib import Path

from alchemi.models.masking import MaskingConfig, make_spatial_mask, make_spectral_mask

__all__ = ["MaskingConfig", "make_spatial_mask", "make_spectral_mask", "persist_mask_config"]


def persist_mask_config(cfg: MaskingConfig, run_dir: Path) -> Path:
    """Compatibility shim to persist mask configs via MaskingConfig.persist."""
    return cfg.persist(run_dir)
