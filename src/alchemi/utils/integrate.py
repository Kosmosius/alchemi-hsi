"""Shared integration helpers that abstract NumPy's trapezoid/trapz rename."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

try:  # NumPy 2.0+
    from numpy import trapezoid as _np_integrate
except ImportError:  # pragma: no cover - NumPy < 2.0 fallback
    from numpy import trapz as _np_integrate

IntegrateFn = Callable[..., Any]
np_integrate: IntegrateFn = _np_integrate

__all__ = ["np_integrate"]
