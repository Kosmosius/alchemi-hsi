"""Shared integration helpers that abstract NumPy's trapezoid/trapz rename."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import numpy as np

IntegrateFn = Callable[..., Any]


def np_integrate(*args: Any, **kwargs: Any) -> Any:
    """Call numpy.trapezoid when available, otherwise fall back to numpy.trapz."""

    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:  # pragma: no cover - NumPy < 2.0 fallback
        trapezoid = np.trapz
    return trapezoid(*args, **kwargs)


__all__ = ["np_integrate"]
