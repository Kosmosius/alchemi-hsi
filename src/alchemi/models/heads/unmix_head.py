from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class LinearUnmixHead(nn.Module):
    def __init__(self, embed_dim: int, k: int = 3) -> None:
        super().__init__()
        self.k = k
        self.fc = nn.Linear(embed_dim, k)

    def forward(self, z: Tensor, basis_emb: Tensor) -> dict[str, Tensor]:
        is_batched = z.dim() > 1
        z_in = z if is_batched else z.unsqueeze(0)

        logits = self.fc(z_in)
        frac = F.softmax(logits, dim=-1)
        recon = (frac.unsqueeze(-1) * basis_emb.unsqueeze(0)).sum(dim=1)

        if not is_batched:
            frac = frac.squeeze(0)
            recon = recon.squeeze(0)

        return {"frac": frac, "recon": recon}
