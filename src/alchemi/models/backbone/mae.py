"""Lightweight masked autoencoder backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from alchemi.config.core import BackboneConfig, ModelConfig
from alchemi.models.blocks import TransformerBlock


def _positional_encoding(num_tokens: int, dim: int, device: torch.device) -> Tensor:
    positions = torch.arange(num_tokens, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device) * (-torch.log(torch.tensor(10000.0)) / dim)
    )
    pe = torch.zeros(num_tokens, dim, device=device)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe


@dataclass
class MAEBackboneOutput:
    encoded: Tensor
    decoded: Tensor
    mask: Tensor


class MAEBackbone(nn.Module):
    """Minimal MAE-style encoder/decoder stack."""

    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        masking_ratio: float = 0.75,
        decoder_dim: int = 512,
        decoder_depth: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.masking_ratio = masking_ratio

        self.encoder_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(depth)]
        )
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_blocks = nn.ModuleList(
            [TransformerBlock(decoder_dim, num_heads) for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.reconstruction = nn.Linear(decoder_dim, embed_dim)

    @classmethod
    def from_config(cls, cfg: ModelConfig | BackboneConfig) -> "MAEBackbone":
        if isinstance(cfg, ModelConfig):
            cfg = cfg.backbone
        return cls(
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            masking_ratio=cfg.masking_ratio,
            decoder_dim=cfg.decoder_dim,
            decoder_depth=cfg.decoder_depth,
        )

    def forward_encoder(self, tokens: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = tokens
        if x.dim() == 2:
            x = x.unsqueeze(0)
        pe = _positional_encoding(x.size(1), x.size(2), x.device)
        x = x + pe
        for block in self.encoder_blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return self.encoder_norm(x)

    def _random_mask(self, num_tokens: int, device: torch.device) -> Tensor:
        keep = int((1.0 - self.masking_ratio) * num_tokens)
        idx = torch.rand(num_tokens, device=device).argsort()
        mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        mask[idx[:keep]] = True
        return mask

    def forward_mae(self, tokens: Tensor) -> MAEBackboneOutput:
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        B, T, D = tokens.shape
        mask = self._random_mask(T, tokens.device)
        visible = tokens[:, mask]
        encoded_visible = self.forward_encoder(visible)

        dec_tokens = self.decoder_embed(encoded_visible)
        mask_token = self.mask_token.expand(B, T - visible.size(1), -1)
        full_dec = torch.empty(B, T, dec_tokens.size(-1), device=tokens.device)
        full_dec[:, mask] = dec_tokens
        full_dec[:, ~mask] = mask_token
        pe = _positional_encoding(T, dec_tokens.size(-1), tokens.device)
        full_dec = full_dec + pe
        for block in self.decoder_blocks:
            full_dec = block(full_dec)
        decoded = self.reconstruction(self.decoder_norm(full_dec))
        return MAEBackboneOutput(encoded=encoded_visible, decoded=decoded, mask=mask)
