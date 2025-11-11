import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self, recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ):
        diff = (recon - target) * (mask.unsqueeze(-1) if mask is not None else 1)
        loss = (diff**2).sum()
        if self.reduction == "mean":
            denom = mask.sum() if mask is not None else torch.tensor(recon.numel(), device=recon.device)
            loss = loss / denom.clamp_min(1)
        return loss
