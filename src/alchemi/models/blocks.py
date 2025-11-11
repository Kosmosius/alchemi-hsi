import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, act=nn.GELU):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), act(), nn.Linear(d_hidden, d_out))

    def forward(self, x):
        return self.net(x)
