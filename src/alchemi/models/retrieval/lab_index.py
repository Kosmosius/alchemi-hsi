"""Nearest-neighbour search over lab spectra embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor


def cosine_topk(query: Tensor, keys: Tensor, k: int = 5) -> tuple[Tensor, Tensor]:
    q = torch.nn.functional.normalize(query.unsqueeze(0), dim=-1)
    kn = torch.nn.functional.normalize(keys, dim=-1)
    sim = (q @ kn.T).squeeze(0)
    vals, idx = torch.topk(sim, min(k, sim.numel()))
    return idx, vals


@dataclass
class LabIndex:
    """Tiny wrapper around cosine search with optional FAISS backend."""

    embeddings: Tensor
    ids: Tensor

    @classmethod
    def build(cls, embeddings: Tensor, ids: Sequence[int] | Tensor | None = None) -> "LabIndex":
        if ids is None:
            ids = torch.arange(embeddings.shape[0], device=embeddings.device)
        elif not isinstance(ids, Tensor):
            ids = torch.tensor(ids, device=embeddings.device)
        return cls(embeddings=embeddings, ids=ids)

    def query_topk(self, embedding: Tensor, k: int = 5) -> tuple[Tensor, Tensor]:
        idx, scores = cosine_topk(embedding, self.embeddings, k)
        return self.ids[idx], scores

    def to(self, device: torch.device) -> "LabIndex":
        self.embeddings = self.embeddings.to(device)
        self.ids = self.ids.to(device)
        return self
