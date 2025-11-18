# mypy: ignore-errors
"""Minimal subset of Hypothesis strategies used in tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = [
    "Strategy",
    "composite",
    "floats",
    "integers",
    "lists",
]


class Strategy:
    def example(self, rng: np.random.Generator):  # pragma: no cover - interface
        raise NotImplementedError


class _CallableStrategy(Strategy):
    def __init__(self, fn: Callable[[np.random.Generator], object]):
        self._fn = fn

    def example(self, rng: np.random.Generator):
        return self._fn(rng)


class _IntegerStrategy(Strategy):
    def __init__(self, min_value: int, max_value: int):
        self.min = int(min_value)
        self.max = int(max_value)

    def example(self, rng: np.random.Generator) -> int:
        if self.min >= self.max:
            return self.min
        return int(rng.integers(self.min, self.max + 1))


class _FloatStrategy(Strategy):
    def __init__(self, min_value: float, max_value: float):
        self.min = float(min_value)
        self.max = float(max_value)

    def example(self, rng: np.random.Generator) -> float:
        if self.min >= self.max:
            return self.min
        return float(rng.uniform(self.min, self.max))


class _ListStrategy(Strategy):
    def __init__(self, element: Strategy, min_size: int, max_size: int):
        self.element = element
        self.min = int(min_size)
        self.max = int(max_size)

    def example(self, rng: np.random.Generator) -> list:
        size = self.min if self.min >= self.max else int(rng.integers(self.min, self.max + 1))
        return [self.element.example(rng) for _ in range(size)]


def integers(*, min_value: int, max_value: int) -> Strategy:
    return _IntegerStrategy(min_value, max_value)


def floats(
    *,
    min_value: float,
    max_value: float,
    allow_nan: bool = False,
    allow_infinity: bool = False,
) -> Strategy:
    return _FloatStrategy(min_value, max_value)


def lists(element: Strategy, *, min_size: int, max_size: int) -> Strategy:
    return _ListStrategy(element, min_size, max_size)


def composite(fn: Callable[..., object]) -> Callable[..., Strategy]:
    def factory(*args, **kwargs) -> Strategy:
        def _example(rng: np.random.Generator):
            def draw(strategy: Strategy):
                if not isinstance(strategy, Strategy):
                    raise TypeError("draw() expects a Strategy instance")
                return strategy.example(rng)

            return fn(draw, *args, **kwargs)

        return _CallableStrategy(_example)

    return factory
