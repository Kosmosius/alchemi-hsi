# mypy: ignore-errors

"""Minimal observability hooks for the Hypothesis pytest plugin."""

# The real plugin tracks test files written during execution to emit warnings.
# We only need to expose the symbol so importing the plugin does not fail.
_WROTE_TO: set[str] = set()

__all__ = ["_WROTE_TO"]
