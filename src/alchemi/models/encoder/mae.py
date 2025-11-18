from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass
class MaskingConfig:
    spatial_mask_ratio: float = 0.75
    spectral_mask_ratio: float = 0.5


class MAEEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, depth: int = 6, n_heads: int = 8) -> None:
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=depth)

    def forward(self, tokens: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
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
