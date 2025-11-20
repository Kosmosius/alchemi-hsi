"""Selective SDP/FlashAttention helpers."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Literal

import torch

SDPBackend = Literal["flash", "mem_efficient", "math"]


@contextmanager
def select_sdp_backend(preference: SDPBackend) -> Iterator[None]:
    """Select the preferred SDPA backend when available.

    On CPU this is a no-op; on CUDA builds with SDPA kernels it toggles the
    flash / memory-efficient / math kernels and restores the original settings
    afterwards.
    """
    if not torch.cuda.is_available() or not hasattr(torch.backends, "cuda"):
        # CPU kernels route through math implementation; nothing to toggle.
        yield
        return

    if not hasattr(torch.backends.cuda, "sdp_kernel"):
        yield
        return

    available = torch.backends.cuda.sdp_kernel
    original = (
        available.is_flash_enabled(),
        available.is_mem_efficient_enabled(),
        available.is_math_enabled(),
    )

    try:
        if preference == "flash":
            available.enable_flash(True)
            available.enable_mem_efficient(True)
            available.enable_math(False)
        elif preference == "mem_efficient":
            available.enable_flash(False)
            available.enable_mem_efficient(True)
            available.enable_math(False)
        else:  # "math"
            available.enable_flash(False)
            available.enable_mem_efficient(False)
            available.enable_math(True)
        yield
    finally:
        available.enable_flash(original[0])
        available.enable_mem_efficient(original[1])
        available.enable_math(original[2])


__all__ = ["select_sdp_backend", "SDPBackend"]
