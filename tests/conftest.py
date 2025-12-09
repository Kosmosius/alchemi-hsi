"""Pytest configuration for the alchemi-hsi test suite."""

from __future__ import annotations

from pathlib import Path
import importlib
import sys

try:  # pragma: no branch - executed in environments with real Hypothesis
    importlib.import_module("hypothesis")
except ModuleNotFoundError:  # pragma: no cover - exercised only in CI without dependency
    stub_module = importlib.import_module("tests._hypothesis_stub")
    HealthCheck = stub_module.HealthCheck
    given = stub_module.given
    settings = stub_module.settings
    _strategies = stub_module.strategies

    # Expose the stub under the public hypothesis namespace so existing imports
    # like ``from hypothesis import given`` continue to work transparently.
    stub = sys.modules.setdefault("hypothesis", sys.modules[__name__])
    stub.HealthCheck = HealthCheck
    stub.given = given
    stub.settings = settings
    stub.strategies = _strategies
    sys.modules.setdefault("hypothesis.strategies", _strategies)


_PHYSICS_METADATA_TARGETS = {
    Path("tests/physics/test_units_and_scales.py"),
    Path("tests/physics/test_planck_roundtrip.py"),
    Path("tests/physics/test_continuum_and_banddepth_against_known_targets.py"),
    Path("tests/physics/test_cube_health_checks.py"),
    Path("tests/spectral/test_srf_validation_and_normalization.py"),
    Path("tests/data/adapters/test_emit_adapter.py"),
    Path("tests/data/adapters/test_enmap_adapter.py"),
    Path("tests/data/adapters/test_avirisng_adapter.py"),
    Path("tests/data/test_adapters_synthetic_units_and_masks.py"),
    Path("tests/data/test_hytes_adapter_physics.py"),
    Path("tests/data/test_mako_ingest.py"),
    Path("tests/data/test_mako_bt_ace.py"),
}


def _is_physics_metadata_only_run(config) -> bool:
    mark_expr = getattr(config.option, "markexpr", "")
    return "physics_and_metadata" in (mark_expr or "")


def pytest_ignore_collect(collection_path: Path, config) -> bool:  # pragma: no cover - pytest hook
    """Optionally skip tests when running with -m physics_and_metadata.

    This hook used to take a ``path`` argument based on :mod:`py.path`.
    Newer versions of pytest pass a :class:`pathlib.Path` ``collection_path``
    instead, and the old argument is deprecated and scheduled for removal.

    To keep the code compatible with both older and newer pytest versions,
    we rely solely on ``collection_path`` here and avoid the deprecated
    ``path`` argument entirely.
    """

    if not _is_physics_metadata_only_run(config):
        return False

    root = Path(config.rootdir).resolve()
    candidate = collection_path.resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        relative = candidate

    if candidate.is_dir():
        return not any(target.is_relative_to(relative) for target in _PHYSICS_METADATA_TARGETS)

    return relative not in _PHYSICS_METADATA_TARGETS
