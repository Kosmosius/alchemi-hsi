"""Pytest configuration for the alchemi-hsi test suite."""

from __future__ import annotations

import importlib
import sys

try:  # pragma: no branch - executed in environments with real Hypothesis
    importlib.import_module("hypothesis")
except ModuleNotFoundError:  # pragma: no cover - exercised only in CI without dependency
    from tests._hypothesis_stub import strategies as _strategies
    from tests._hypothesis_stub import (  # noqa: F401 - re-exported via sys.modules
        HealthCheck,
        given,
        settings,
    )

    # Expose the stub under the public hypothesis namespace so existing imports
    # like ``from hypothesis import given`` continue to work transparently.
    stub = sys.modules.setdefault("hypothesis", sys.modules[__name__])
    stub.HealthCheck = HealthCheck
    stub.given = given
    stub.settings = settings
    stub.strategies = _strategies
    sys.modules.setdefault("hypothesis.strategies", _strategies)
