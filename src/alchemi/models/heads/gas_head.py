import torch
import torch.nn as nn


class GasHead(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)
