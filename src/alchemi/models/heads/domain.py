import torch
import torch.nn as nn


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


class DomainDiscriminator(nn.Module):
    """Optional domain head for sensor confusion."""

    def __init__(self, embed_dim: int, n_domains: int):
        super().__init__()
        self.fe = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ReLU())
        self.clf = nn.Linear(embed_dim // 2, n_domains)
        self.lambd = 0.0

    def set_lambda(self, lambd: float):
        self.lambd = lambd

    def forward(self, z):
        z = GradReverse.apply(z, self.lambd)
        return self.clf(self.fe(z))
