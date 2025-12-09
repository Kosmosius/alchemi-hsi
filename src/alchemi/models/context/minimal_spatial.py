"""Minimal spatial context using lightweight CNN layers."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from alchemi.models.blocks import MLP


class MinimalSpatialContext(nn.Module):
    """Extracts local spatial embeddings and fuses them with per-pixel features."""

    def __init__(self, embed_dim: int, k: int = 5, hidden_mult: float = 2.0) -> None:
        super().__init__()
        padding = k // 2
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=k, padding=padding, groups=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AvgPool2d(kernel_size=k, stride=1, padding=padding)
        self.fuse = MLP(embed_dim * 2, int(embed_dim * hidden_mult), embed_dim)

    def forward(self, features: Tensor) -> Tensor:
        """Fuse spatial context.

        Parameters
        ----------
        features:
            Tensor shaped ``(B, H, W, C)`` coming from the backbone.
        """

        if features.dim() != 4:
            raise ValueError("Expected features with shape (B, H, W, C)")
        b, h, w, c = features.shape
        x = features.permute(0, 3, 1, 2)
        local = self.conv(x)
        pooled = self.pool(local)
        fused = torch.cat([local, pooled], dim=1)
        fused = fused.permute(0, 2, 3, 1).reshape(b, h, w, -1)
        return self.fuse(fused)
