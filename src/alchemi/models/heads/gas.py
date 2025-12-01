"""Gas detection head using spectral windows and spatial context."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from alchemi.config.core import GasHeadConfig, ModelConfig
from alchemi.models.blocks import MLP
from alchemi.models.context import MinimalSpatialContext


@dataclass
class GasOutput:
    enhancement_mean: Tensor
    enhancement_logvar: Tensor
    plume_logits: Tensor


class GasHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 512,
        *,
        spectral_indices: Sequence[int] | None = None,
        spatial_context: bool = True,
    ) -> None:
        super().__init__()
        self.spectral_indices = list(spectral_indices) if spectral_indices is not None else None
        self.context = MinimalSpatialContext(embed_dim) if spatial_context else None
        self.enhancement_head = MLP(embed_dim, hidden_dim, 2)
        self.plume_head = MLP(embed_dim, hidden_dim, 1)

    @classmethod
    def from_config(
        cls,
        embed_dim: int,
        cfg: ModelConfig | GasHeadConfig,
        *,
        spectral_indices: Sequence[int] | None = None,
    ) -> "GasHead":
        if isinstance(cfg, ModelConfig):
            cfg = cfg.heads.gas
        return cls(
            embed_dim=embed_dim,
            hidden_dim=cfg.hidden_dim,
            spectral_indices=spectral_indices,
        )

    def _select_spectral(self, features: Tensor) -> Tensor:
        if self.spectral_indices is None:
            return features
        return features[..., self.spectral_indices]

    def forward(self, features: Tensor) -> GasOutput:
        if features.dim() != 4:
            raise ValueError("GasHead expects features shaped (B, H, W, C)")
        x = self._select_spectral(features)
        if self.context is not None:
            x = self.context(x)
        flat = x.view(-1, x.shape[-1])
        enhancement = self.enhancement_head(flat)
        plume = self.plume_head(flat)
        mean, logvar = enhancement.split(1, dim=-1)
        return GasOutput(
            enhancement_mean=mean.view(*features.shape[:3], -1),
            enhancement_logvar=logvar.view(*features.shape[:3], -1),
            plume_logits=plume.view(*features.shape[:3], -1),
        )
