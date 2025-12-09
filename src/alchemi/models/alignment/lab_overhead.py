"""Alignment utilities for lab and overhead spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor, nn

from alchemi.config.core import AlignmentConfig, ModelConfig
from alchemi.models.blocks import MLP


@dataclass
class AlignmentOutputs:
    loss: Tensor
    contrastive: Tensor
    cycle: Tensor
    logits: Tensor


class LabOverheadAlignment(nn.Module):
    """Projection heads and lightweight cycle-consistency helpers."""

    def __init__(
        self,
        embed_dim: int = 256,
        projection_dim: int = 256,
        depth: int = 2,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        hidden = max(embed_dim, projection_dim)
        layers = []
        for _ in range(depth - 1):
            layers.append(MLP(embed_dim, hidden, embed_dim))
            embed_dim = hidden
        layers.append(nn.Linear(embed_dim, projection_dim))
        self.lab_projector = nn.Sequential(*layers)
        self.overhead_projector = nn.Sequential(*layers)
        self.lab_decoder = MLP(projection_dim, hidden, projection_dim)
        self.overhead_decoder = MLP(projection_dim, hidden, projection_dim)

    @classmethod
    def from_config(cls, cfg: ModelConfig | AlignmentConfig) -> "LabOverheadAlignment":
        if isinstance(cfg, ModelConfig):
            cfg = cfg.alignment
        return cls(
            projection_dim=cfg.projection_dim,
            temperature=cfg.temperature,
            depth=cfg.projection_depth,
        )

    def project_lab(self, embedding: Tensor) -> Tensor:
        return self.lab_projector(embedding)

    def project_overhead(self, embedding: Tensor) -> Tensor:
        return self.overhead_projector(embedding)

    def cycle_lab(self, overhead_embedding: Tensor) -> Tensor:
        return self.lab_decoder(self.project_overhead(overhead_embedding))

    def cycle_overhead(self, lab_embedding: Tensor) -> Tensor:
        return self.overhead_decoder(self.project_lab(lab_embedding))

    def forward(self, lab_embeddings: Tensor, overhead_embeddings: Tensor) -> AlignmentOutputs:
        return alignment_losses(
            lab_embeddings,
            overhead_embeddings,
            self.temperature,
            self.project_lab,
            self.project_overhead,
            self.cycle_lab,
            self.cycle_overhead,
        )


def alignment_losses(
    lab_embeddings: Tensor,
    overhead_embeddings: Tensor,
    temperature: float = 0.07,
    project_lab: Optional[Callable[[Tensor], Tensor]] = None,
    project_overhead: Optional[Callable[[Tensor], Tensor]] = None,
    cycle_lab: Optional[Callable[[Tensor], Tensor]] = None,
    cycle_overhead: Optional[Callable[[Tensor], Tensor]] = None,
) -> AlignmentOutputs:
    """Compute InfoNCE-style contrastive and optional cycle losses."""

    z_lab = project_lab(lab_embeddings) if project_lab is not None else lab_embeddings
    z_ovh = (
        project_overhead(overhead_embeddings)
        if project_overhead is not None
        else overhead_embeddings
    )

    z_lab = torch.nn.functional.normalize(z_lab, dim=-1)
    z_ovh = torch.nn.functional.normalize(z_ovh, dim=-1)

    logits = z_lab @ z_ovh.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    contrastive = nn.functional.cross_entropy(logits, labels)
    contrastive = contrastive + nn.functional.cross_entropy(logits.t(), labels)

    cycle_loss = torch.tensor(0.0, device=logits.device)
    if cycle_lab is not None:
        recon_lab = cycle_lab(z_ovh)
        cycle_loss = cycle_loss + nn.functional.mse_loss(recon_lab, z_lab)
    if cycle_overhead is not None:
        recon_ovh = cycle_overhead(z_lab)
        cycle_loss = cycle_loss + nn.functional.mse_loss(recon_ovh, z_ovh)

    total = contrastive + cycle_loss
    return AlignmentOutputs(loss=total, contrastive=contrastive, cycle=cycle_loss, logits=logits)
