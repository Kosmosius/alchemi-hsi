from __future__ import annotations

import torch
import torch.nn as nn


class SpectralSmoothnessLoss(nn.Module):
    """Second-derivative smoothness penalty along the spectral axis.

    Expects spectra in the last dimension. An optional boolean mask can be
    provided to ignore invalid bands.
    """

    def forward(self, y: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # First-order differences along the spectral dimension
        d1 = y[..., 1:] - y[..., :-1]

        if mask is not None:
            # Only penalise where both neighbours are valid
            m = mask[..., 1:] & mask[..., :-1]
            # Broadcast mask to match d1's dimensionality
            while m.ndim < d1.ndim:
                m = m.unsqueeze(-1)
            d1 = d1 * m.float()

        # Second-order differences
        d2 = d1[..., 1:] - d1[..., :-1]
        return (d2**2).mean()
