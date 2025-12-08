"""Compatibility shim for auxiliary prediction heads.

This module exposes the public auxiliary head API under
``alchemi.models.heads.aux`` while reusing the implementation in
``aux_head.py``. Keep the implementation in ``aux_head.py`` and import
from here to converge on a single module path.
"""

from .aux_head import AuxHead, AuxOutputs

__all__ = ["AuxHead", "AuxOutputs"]
