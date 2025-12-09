"""Factory helpers for constructing common encoder blocks.

These helpers are intentionally lightweight and avoid any training-specific
assumptions. They exist so both the synthetic MAE pretrainer and the CLIP-style
alignment trainer can share identical encoder plumbing without diverging over
time.
"""

from __future__ import annotations

from .set_encoder import SetEncoder


def build_set_encoder(*, embed_dim: int, depth: int, heads: int) -> SetEncoder:
    """Create the canonical SetEncoder used by both training stacks."""

    return SetEncoder(dim=embed_dim, depth=depth, heads=heads)
