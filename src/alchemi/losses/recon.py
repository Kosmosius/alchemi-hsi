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
        diff = recon - target
        if mask is not None:
            mask_f = mask.to(recon.device, recon.dtype)
            while mask_f.dim() < diff.dim():
                mask_f = mask_f.unsqueeze(-1)
            diff = diff * mask_f
            denom = mask_f.sum() * diff.shape[-1]
        else:
            denom = torch.tensor(diff.numel(), device=recon.device, dtype=recon.dtype)

        loss = (diff**2).sum()
        if self.reduction == "mean":
            loss = loss / denom.clamp_min(1)
        return loss
