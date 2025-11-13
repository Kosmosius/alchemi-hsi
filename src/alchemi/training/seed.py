import os
import random

import numpy as np
import torch

_NP_GENERATOR: np.random.Generator | None = None


def seed_everything(seed: int = 42):
    global _NP_GENERATOR
    random.seed(seed)
    _NP_GENERATOR = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
