"""Reusable neural network blocks used across the ALCHEMI model stack.

The implementations here intentionally stay lightweight and torch-native so the
components can be swapped with more optimised variants (e.g. FlashAttention)
without changing call-sites. Shapes follow the PyTorch transformer defaults
(batch-first).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

__all__ = [
    "MLP",
    "MultiHeadSelfAttention",
    "TransformerBlock",
]


class MLP(nn.Module):
    """Two-layer feed-forward projection with configurable activation."""

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        *,
        act: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            act(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        """Apply the MLP."""
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    """Thin wrapper over :class:`nn.MultiheadAttention` with batch-first inputs."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

    def forward(
        self, x: Tensor, key_padding_mask: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """Compute self-attention for ``x`` with an optional padding mask."""

        y, weights = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=True)
        return y, weights


class TransformerBlock(nn.Module):
    """Standard transformer encoder block (attention + MLP)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden, dim, dropout=dropout)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Apply attention and feed-forward residual blocks."""

        attn_out, _ = self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
