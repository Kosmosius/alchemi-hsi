from __future__ import annotations

import torch
from torch import Tensor, nn


class ReconstructionLoss(nn.Module):
    """Mask-aware MSE reconstruction loss.

    - If mask is None: plain MSE over all elements.
    - If mask is provided: it is broadcast to recon/target and only masked
      locations contribute to the loss.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction

    def forward(
        self,
        recon: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        diff = recon - target
        if mask is not None:
            # Broadcast mask up to recon/target rank by unsqueezing trailing dims.
            mask_expanded = mask.to(device=recon.device, dtype=recon.dtype)
            while mask_expanded.dim() < diff.dim():
                mask_expanded = mask_expanded.unsqueeze(-1)
            diff = diff * mask_expanded
            denom = mask_expanded.sum()
        else:
            denom = torch.tensor(diff.numel(), device=recon.device, dtype=recon.dtype)

        loss = (diff**2).sum()
        if self.reduction == "mean":
            loss = loss / denom.clamp_min(1)
        return loss
