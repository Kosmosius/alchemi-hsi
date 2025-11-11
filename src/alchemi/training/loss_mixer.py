from dataclasses import dataclass


@dataclass
class Weights:
    recon: float = 1.0
    nce: float = 1.0
    sam: float = 0.0
    smooth: float = 0.0
