import torch


class SpectralNoise:
    def __init__(self, sigma: float = 0.002):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.sigma * torch.randn_like(x)


class RandomBandDropout:
    def __init__(self, p: float = 0.05):
        self.p = p

    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        drop = (torch.rand_like(mask.float()) < self.p).bool()
        return mask & ~drop
