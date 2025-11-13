"""Bootstrap tests for the alchemi_hsi package."""

from __future__ import annotations

import importlib
import re

import alchemi_hsi


SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


def test_package_importable() -> None:
    module = importlib.import_module("alchemi_hsi")
    assert module is alchemi_hsi


def test_version_is_semver() -> None:
    assert SEMVER_PATTERN.match(alchemi_hsi.__version__) is not None


def test_dunder_all_contains_version() -> None:
    assert "__version__" in alchemi_hsi.__all__
