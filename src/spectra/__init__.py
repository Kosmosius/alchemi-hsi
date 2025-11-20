"""Spectral masked-autoencoder utilities and DDP/precision helpers.

This package bundles lightweight data modules, masking-aware tokenizers, and
training loops used throughout the tests.  The public surface keeps backward
compatibility with the existing tokenizer/positional-encoding helpers while
adding the richer MAE components referenced in the PRD.
"""

from . import config, data, models, train, utils
from .models import SpectralTokenizer, WavelengthPosEnc, WavelengthPositionalEncoding

__all__ = [
    "config",
    "data",
    "models",
    "train",
    "utils",
    "SpectralTokenizer",
    "WavelengthPosEnc",
    "WavelengthPositionalEncoding",
]
