from __future__ import annotations

import torch
from torch import Tensor, nn


class ReconstructionLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        recon: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        diff = (recon - target) * (mask.unsqueeze(-1) if mask is not None else 1)
        loss = (diff**2).sum()
        if self.reduction == "mean":
            denom = (
                mask.sum() if mask is not None else torch.tensor(recon.numel(), device=recon.device)
            )
            loss = loss / denom.clamp_min(1)
        return loss
