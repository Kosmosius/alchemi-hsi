# mypy: ignore-errors
"""Minimal stub of the Hypothesis API used in tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, TypeVar, overload, cast

import numpy as _np

from . import strategies  # re-exported module
from .strategies import Strategy

__all__ = ["HealthCheck", "given", "settings", "strategies"]


F = TypeVar("F", bound=Callable[..., Any])


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


@overload
def given(fn: F, /) -> F: ...


@overload
def given(*strategies: Strategy) -> Callable[[F], F]: ...


def given(*args: Any) -> Callable[[F], F] | F:
    """Execute the wrapped test with randomly sampled arguments."""

    if args and callable(args[0]):
        fn = cast(F, args[0])
        return _wrap_with_strategies(fn, ())

    strategies = tuple(cast(tuple[Strategy, ...], args))

    def decorator(fn: F) -> F:
        return _wrap_with_strategies(fn, strategies)

    return decorator


def _wrap_with_strategies(fn: F, strategies: tuple[Strategy, ...]) -> F:
    if not strategies:
        return fn

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        cfg = getattr(fn, "__hypothesis_settings__", {})
        max_examples = int(cfg.get("max_examples", 10))
        for _ in range(max_examples):
            samples = [strategy.example(_GLOBAL_RNG) for strategy in strategies]
            fn(*args, *samples, **kwargs)

    wrapper.__name__ = getattr(fn, "__name__", getattr(wrapper, "__name__", "wrapper"))
    return cast(F, wrapper)
