"""Minimal stub of the Hypothesis API used in tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as _np

from . import strategies  # re-exported module
from .strategies import Strategy

__all__ = ["HealthCheck", "given", "settings", "strategies"]


@dataclass(frozen=True)
class _HealthCheck:
    too_slow: str = "too_slow"


HealthCheck = _HealthCheck()


def settings(
    *,
    max_examples: int = 10,
    deadline: float | None = None,
    suppress_health_check: Iterable[str] | None = None,
):
    """Store lightweight settings metadata on the wrapped function."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        fn.__hypothesis_settings__ = {"max_examples": int(max_examples)}
        return fn

    return decorator


_GLOBAL_RNG = _np.random.default_rng(0)


def given(*strategies: Strategy) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Execute the wrapped test with randomly sampled arguments."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if not strategies:
            raise TypeError("given() requires at least one strategy")

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cfg = getattr(fn, "__hypothesis_settings__", {})
            max_examples = int(cfg.get("max_examples", 10))
            for _ in range(max_examples):
                samples = [strategy.example(_GLOBAL_RNG) for strategy in strategies]
                fn(*args, *samples, **kwargs)

        wrapper.__name__ = getattr(fn, "__name__", wrapper.__name__)
        return wrapper

    return decorator


