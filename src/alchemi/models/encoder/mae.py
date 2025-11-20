from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class MaskingConfig:
    spatial_mask_ratio: float = 0.75
    spectral_mask_ratio: float = 0.5
    no_spatial_mask: bool = False
    no_posenc: bool = False


class MAEEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 6,
        n_heads: int = 8,
        *,
        use_posenc: bool = True,
        posenc: nn.Module | None = None,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=depth)
        self.use_posenc = use_posenc
        self.posenc = None if not use_posenc else posenc or nn.Embedding(max_tokens, embed_dim)

    def forward(self, tokens: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        if self.use_posenc and self.posenc is not None:
            positions = torch.arange(tokens.size(1), device=tokens.device)
            pos = self.posenc(positions)
            if pos.dim() == 2:
                pos = pos.unsqueeze(0)
            tokens = tokens + pos[:, : tokens.size(1), :]
        return self.enc(tokens, src_key_padding_mask=key_padding_mask)


class MAEDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 4,
        n_heads: int = 8,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        dec = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True)
        self.dec = nn.TransformerDecoder(dec, num_layers=depth)
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, z: Tensor, mem: Tensor | None = None) -> Tensor:
        if mem is None:
            mem = z
        y = self.dec(z, mem)
        return self.proj(y)
