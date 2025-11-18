from __future__ import annotations

import torch
from torch import Tensor, nn


class SpectralBasisProjector(nn.Module):
    """
    Projects irregular bands to fixed K-dim vector using wavelength RBF bases.
    Input per sample: wavelengths [B], values [B], mask [B]
    Output: phi [K]
    """

    def __init__(
        self,
        K: int = 128,
        lambda_min: float = 350.0,
        lambda_max: float = 12000.0,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        centers = torch.linspace(lambda_min, lambda_max, K)
        scales = (lambda_max - lambda_min) / K
        self.centers = nn.Parameter(centers, requires_grad=learnable)
        self.log_bw = nn.Parameter(
            torch.full((K,), torch.log(torch.tensor(scales))), requires_grad=learnable
        )

    def forward(self, wavelengths: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        lam = wavelengths.unsqueeze(-1)
        ctr = self.centers.unsqueeze(0).to(lam)
        bw = torch.exp(self.log_bw).unsqueeze(0).to(lam)
        rbf = torch.exp(-0.5 * ((lam - ctr) / (bw + 1e-9)) ** 2)
        rbf = rbf * mask.float().unsqueeze(-1)
        num = (rbf * values.unsqueeze(-1)).sum(dim=0)
        den = rbf.sum(dim=0).clamp_min(1e-8)
        return num / den
