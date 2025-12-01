"""Solids head leveraging prototype retrieval."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from alchemi.config.core import ModelConfig, SolidsHeadConfig
from alchemi.models.blocks import MLP
from alchemi.models.retrieval import LabIndex


@dataclass
class SolidsOutput:
    dominant_id: Tensor
    abundances: Tensor
    reconstruction: Tensor
    scores: Tensor


class SolidsHead(nn.Module):
    """Predicts material abundances from per-pixel embeddings."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 512,
        k: int = 5,
        *,
        prototype_spectra: Optional[Tensor] = None,
        lab_index: Optional[LabIndex] = None,
    ) -> None:
        super().__init__()
        self.k = k
        self.lab_index = lab_index
        self.prototype_spectra = prototype_spectra
        self.score_proj = MLP(embed_dim, hidden_dim, k)
        self.abundance_proj = MLP(embed_dim + k, hidden_dim, k + 2)

    @classmethod
    def from_config(
        cls,
        embed_dim: int,
        cfg: ModelConfig | SolidsHeadConfig,
        *,
        lab_index: Optional[LabIndex] = None,
        prototype_spectra: Optional[Tensor] = None,
    ) -> "SolidsHead":
        if isinstance(cfg, ModelConfig):
            cfg = cfg.heads.solids
        return cls(
            embed_dim=embed_dim,
            hidden_dim=cfg.hidden_dim,
            k=5,
            lab_index=lab_index,
            prototype_spectra=prototype_spectra,
        )

    def forward(self, features: Tensor) -> SolidsOutput:
        if features.dim() == 4:
            b, h, w, c = features.shape
            flat = features.view(-1, c)
        else:
            b, h, w = features.shape[0], 1, 1
            flat = features

        if self.lab_index is None:
            raise ValueError("LabIndex must be provided to query prototypes")

        scores = self.score_proj(flat)
        topk_ids = []
        topk_sims = []
        with torch.no_grad():
            for vec in flat:
                ids, sims = self.lab_index.query_topk(vec, self.k)
                topk_ids.append(ids)
                topk_sims.append(sims)
        cand_ids = torch.stack(topk_ids)
        cand_sims = torch.stack(topk_sims)
        fused = torch.cat([flat, scores], dim=-1)
        abundance_logits = self.abundance_proj(fused)
        abundances = torch.nn.functional.softmax(abundance_logits, dim=-1)

        if self.prototype_spectra is not None:
            protos = self.prototype_spectra[cand_ids]
            recon = (abundances[..., : self.k].unsqueeze(-1) * protos).sum(dim=1)
        else:
            recon = torch.zeros(flat.shape[0], 1, device=flat.device, dtype=flat.dtype)

        dominant = cand_ids[torch.arange(cand_ids.shape[0]), abundances[:, : self.k].argmax(dim=-1)]
        output = SolidsOutput(
            dominant_id=dominant.view(b, h, w),
            abundances=abundances.view(b, h, w, -1),
            reconstruction=recon.view(b, h, w, -1),
            scores=cand_sims.view(b, h, w, -1),
        )
        return output
