"""Auxiliary prediction heads for band depth and QA."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from alchemi.config.core import AuxHeadConfig, ModelConfig
from alchemi.models.blocks import MLP


@dataclass
class AuxOutputs:
    band_depth: Tensor
    qa_logits: Tensor


class AuxHead(nn.Module):
    def __init__(self, embed_dim: int, projection_dim: int = 256) -> None:
        super().__init__()
        self.band_depth = MLP(embed_dim, projection_dim, 1)
        self.qa = MLP(embed_dim, projection_dim, 2)

    @classmethod
    def from_config(cls, embed_dim: int, cfg: ModelConfig | AuxHeadConfig) -> "AuxHead":
        if isinstance(cfg, ModelConfig):
            cfg = cfg.heads.aux
        return cls(embed_dim=embed_dim, projection_dim=cfg.projection_dim)

    def forward(self, features: Tensor) -> AuxOutputs:
        flat = features.view(-1, features.shape[-1])
        depth = self.band_depth(flat)
        qa = self.qa(flat)
        return AuxOutputs(
            band_depth=depth.view(*features.shape[:-1], -1),
            qa_logits=qa.view(*features.shape[:-1], -1),
        )
