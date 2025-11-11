import torch
import torch.nn as nn


class SpectralSmoothnessLoss(nn.Module):
    def forward(self, y: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        d1 = y[..., 1:] - y[..., :-1]
        if mask is not None:
            m = mask[..., 1:] & mask[..., :-1]
            d1 = d1 * m.unsqueeze(-1).float() if d1.dim() > 1 else d1 * m.float()
        d2 = d1[..., 1:] - d1[..., :-1]
        return (d2**2).mean()
