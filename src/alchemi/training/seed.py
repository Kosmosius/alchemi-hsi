from __future__ import annotations

import numpy as np
from spectra.utils.seed import seed_everything as _spectra_seed

_NP_GENERATOR: np.random.Generator | None = None


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """Seed randomness across Python, NumPy, and PyTorch.

    Delegates to :func:`spectra.utils.seed.seed_everything` to ensure consistent
    behavior across distributed workers and optionally configure deterministic
    settings. The NumPy generator is cached for callers that rely on a
    reusable RNG instance.
    """
    global _NP_GENERATOR
    _NP_GENERATOR = _spectra_seed(seed=seed, deterministic=deterministic)
