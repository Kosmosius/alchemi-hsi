"""Collection of lightweight loss helpers used by the trainer."""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from alchemi.losses import InfoNCELoss
from alchemi.models.heads.aux_head import AuxOutputs  # NOTE: aux head lives in aux_head.py (Windows `AUX` reserved)
from alchemi.models.heads.gas import GasOutput
from alchemi.models.heads.solids import SolidsOutput

Tensor = torch.Tensor


def build_mae_reconstruction_loss(
    sam_weight: float = 0.0, derivative_weight: float = 0.0
) -> Callable[[Tensor, Tensor, Tensor | None], Tensor]:
    """MSE reconstruction with optional spectral angle and derivative regularisation."""

    def loss_fn(pred: Tensor, target: Tensor, *, mask: Tensor | None = None) -> Tensor:
        if mask is not None:
            pred = pred[:, mask]
            target = target[:, mask]
        mse = F.mse_loss(pred, target)
        loss = mse
        if sam_weight > 0:
            norm_pred = F.normalize(pred, dim=-1)
            norm_target = F.normalize(target, dim=-1)
            sam = torch.acos((norm_pred * norm_target).sum(dim=-1).clamp(-1.0, 1.0)).mean()
            loss = loss + sam_weight * sam
        if derivative_weight > 0 and target.size(-1) > 1:
            deriv = F.mse_loss(pred[..., 1:] - pred[..., :-1], target[..., 1:] - target[..., :-1])
            loss = loss + derivative_weight * deriv
        return loss

    return loss_fn


def build_info_nce_loss(temperature: float = 0.07) -> InfoNCELoss:
    return InfoNCELoss(temperature)


def build_solids_loss() -> Callable[[SolidsOutput, Tensor | None], Tensor]:
    def loss_fn(output: SolidsOutput, labels: Tensor | None) -> Tensor:
        if labels is None:
            return torch.zeros((), device=output.abundances.device)
        # assume labels contains abundance targets matching last dim
        labels = labels.to(output.abundances.device)
        labels = labels.view_as(output.abundances[..., 0])
        dominant = output.abundances[..., 0]
        return F.l1_loss(dominant, labels.float())

    return loss_fn


def build_gas_loss() -> Callable[[GasOutput, Tensor | None], Tensor]:
    def loss_fn(output: GasOutput, labels: Tensor | None) -> Tensor:
        if labels is None:
            return torch.zeros((), device=output.enhancement_mean.device)
        labels = labels.to(output.enhancement_mean.device)
        mean = output.enhancement_mean.squeeze(-1)
        logvar = output.enhancement_logvar.squeeze(-1)
        var = logvar.exp()
        nll = 0.5 * ((labels - mean) ** 2 / var + logvar)
        if output.plume_logits is not None:
            plume = F.binary_cross_entropy_with_logits(output.plume_logits.squeeze(-1), (labels > 0).float())
            return nll.mean() + plume
        return nll.mean()

    return loss_fn


def build_aux_loss() -> Callable[[AuxOutputs, Tensor | None], Tensor]:
    def loss_fn(output: AuxOutputs, targets: Tensor | None) -> Tensor:
        if targets is None:
            return torch.zeros((), device=output.band_depth.device)
        targets = targets.to(output.band_depth.device)
        targets = targets.view_as(output.band_depth)
        depth_loss = F.l1_loss(output.band_depth, targets)
        qa = torch.zeros_like(output.qa_logits)
        qa_loss = F.cross_entropy(output.qa_logits.view(-1, output.qa_logits.size(-1)), qa.view(-1))
        return depth_loss + qa_loss

    return loss_fn


__all__ = [
    "build_mae_reconstruction_loss",
    "build_info_nce_loss",
    "build_solids_loss",
    "build_gas_loss",
    "build_aux_loss",
]
