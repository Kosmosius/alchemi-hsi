import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMLoss(nn.Module):
    def forward(
        self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is not None:
            x = x * mask.unsqueeze(-1)
            y = y * mask.unsqueeze(-1)
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        cos = (x * y).sum(dim=-1).clamp(-1, 1)
        ang = torch.arccos(cos)
        return ang.mean()
