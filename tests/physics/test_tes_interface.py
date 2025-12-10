"""Placeholder contract tests for the TES API."""

import pytest

from alchemi.physics import tes


def test_tes_lwirt_is_stub_with_forward_compatibility_message():
    with pytest.raises(NotImplementedError) as excinfo:
        tes.tes_lwirt(L=None)  # type: ignore[arg-type]

    message = str(excinfo.value)
    assert "research-grade" in message
    assert "not implemented" in message
    assert "v1.1" in message
