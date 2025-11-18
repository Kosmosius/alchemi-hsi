from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class LinearUnmixHead(nn.Module):
    def __init__(self, embed_dim: int, k: int = 3) -> None:
        super().__init__()
        self.k = k
        self.fc = nn.Linear(embed_dim, k)

    def forward(self, z: Tensor, basis_emb: Tensor) -> dict[str, Tensor]:
        logits = self.fc(z)
        frac = F.softmax(logits, dim=-1)
        recon = (frac.unsqueeze(-1) * basis_emb).sum(dim=0)
        return {"frac": frac, "recon": recon}
