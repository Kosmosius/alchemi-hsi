from __future__ import annotations

from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        act: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), act(), nn.Linear(d_hidden, d_out))

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401 - simple forward description
        return self.net(x)
