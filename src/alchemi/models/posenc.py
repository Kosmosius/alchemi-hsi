import torch
import torch.nn as nn


class WavelengthPosEnc(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

    def forward(self, wavelengths_nm: torch.Tensor) -> torch.Tensor:
        w = wavelengths_nm.unsqueeze(-1)
        div = torch.exp(
            torch.arange(0, self.dim, 2, device=w.device, dtype=w.dtype)
            * (-torch.log(torch.tensor(10000.0)) / self.dim)
        )
        return torch.cat([torch.sin(w * div), torch.cos(w * div)], dim=-1)
