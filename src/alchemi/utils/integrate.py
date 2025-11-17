"""Shared integration helpers that abstract NumPy's trapezoid/trapz rename."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

IntegrateFn = Callable[..., Any]
np_integrate: IntegrateFn = getattr(np, "trapezoid", np.trapz)

__all__ = ["np_integrate"]
