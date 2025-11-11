import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearUnmixHead(nn.Module):
    def __init__(self, embed_dim: int, k: int = 3):
        super().__init__()
        self.k = k
        self.fc = nn.Linear(embed_dim, k)

    def forward(self, z: torch.Tensor, basis_emb: torch.Tensor):
        logits = self.fc(z)
        frac = F.softmax(logits, dim=-1)
        recon = (frac.unsqueeze(-1) * basis_emb).sum(dim=0)
        return {"frac": frac, "recon": recon}
