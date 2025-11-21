from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class MaskingConfig:
    spatial_mask_ratio: float = 0.5
    spectral_mask_ratio: float = 0.5
    mask_seed: int = 1234
    spectral_grouping: int | None = None


class MaskingHelper(nn.Module):
    def __init__(self, config: MaskingConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("_seed_tensor", torch.tensor(config.mask_seed), persistent=False)

    def _rng(self) -> torch.Generator:
        g = torch.Generator(device=self._seed_tensor.device)
        g.manual_seed(int(self._seed_tensor.item()))
        self._seed_tensor.add_(1)
        return g

    def spatial_mask(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, num_tokens, _ = tokens.shape
        keep = max(1, int((1 - self.config.spatial_mask_ratio) * num_tokens))
        rng = self._rng()
        idx = torch.rand(batch, num_tokens, generator=rng, device=tokens.device).argsort(dim=1)
        keep_idx = idx[:, :keep]
        mask = torch.ones(batch, num_tokens, device=tokens.device, dtype=torch.bool)
        mask.scatter_(1, keep_idx, False)
        return mask, keep_idx

    def spectral_mask(self, band_mask: torch.Tensor) -> torch.Tensor:
        batch, bands = band_mask.shape
        rng = self._rng()
        keep = 1 - self.config.spectral_mask_ratio
        probs = torch.rand(batch, bands, generator=rng, device=band_mask.device)
        spec_mask = probs > keep
        spec_mask = torch.where(band_mask, spec_mask, torch.ones_like(spec_mask))
        return spec_mask

    def combined_mask(
        self, tokens: torch.Tensor, band_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        spatial_mask, keep_idx = self.spatial_mask(tokens)
        spectral_mask = self.spectral_mask(band_mask)
        return {
            "spatial_mask": spatial_mask,
            "spatial_keep_idx": keep_idx,
            "spectral_mask": spectral_mask,
        }


def make_spatial_mask(tokens: torch.Tensor, cfg: MaskingConfig) -> torch.Tensor:
    helper = MaskingHelper(cfg)
    spatial_mask, _ = helper.spatial_mask(tokens)
    return spatial_mask


def make_spectral_mask(band_mask: torch.Tensor, cfg: MaskingConfig) -> torch.Tensor:
    helper = MaskingHelper(cfg)
    return helper.spectral_mask(band_mask)


def persist_mask_config(cfg: MaskingConfig, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "mask_config.json"
    import json

    path.write_text(json.dumps(cfg.__dict__, indent=2))
    return path


__all__ = [
    "MaskingConfig",
    "MaskingHelper",
    "make_spatial_mask",
    "make_spectral_mask",
    "persist_mask_config",
]
