from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, lambd: float) -> Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor, None]:
        return -ctx.lambd * grad, None


class DomainDiscriminator(nn.Module):
    """Optional domain head for sensor confusion."""

    def __init__(self, embed_dim: int, n_domains: int) -> None:
        super().__init__()
        self.fe = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ReLU())
        self.clf = nn.Linear(embed_dim // 2, n_domains)
        self.lambd = 0.0

    def set_lambda(self, lambd: float) -> None:
        self.lambd = lambd

    def forward(self, z: Tensor) -> Tensor:
        z = GradReverse.apply(z, self.lambd)
        return self.clf(self.fe(z))
