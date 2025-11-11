import torch
import torch.nn as nn


class SetEncoder(nn.Module):
    """
    Lightweight set transformer for variable-length band tokens.
    Input tokens per sample: x [B, D]; mask [B] (False for padding)
    Returns pooled embedding [D]
    """

    def __init__(self, dim: int = 256, depth: int = 2, heads: int = 4):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=depth)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        padding_mask = ~mask
        out = self.enc(tokens.unsqueeze(0), src_key_padding_mask=padding_mask.unsqueeze(0))
        pooled = out[:, mask, :].mean(dim=1) if mask.any() else out.mean(dim=1)
        return pooled.squeeze(0)
