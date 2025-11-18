from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def cosine_topk(query: Tensor, keys: Tensor, k: int = 5) -> tuple[Tensor, Tensor]:
    q = F.normalize(query.unsqueeze(0), dim=-1)
    kn = F.normalize(keys, dim=-1)
    sim = (q @ kn.T).squeeze(0)
    vals, idx = torch.topk(sim, min(k, sim.numel()))
    return idx, vals
