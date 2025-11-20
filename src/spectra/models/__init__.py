"""Spectral models: tokenization, wavelength encodings, and masked autoencoding."""

from .mae import MAEOutput, MAEDecoder, MAEEncoder, MaskedAutoencoder
from .masking import MaskingConfig, make_spatial_mask, make_spectral_mask, persist_mask_config
from .posenc import WavelengthPosEnc
from .tokenizer import SpectralTokenizer

__all__ = [
    # MAE + masking
    "MAEOutput",
    "MAEDecoder",
    "MAEEncoder",
    "MaskedAutoencoder",
    "MaskingConfig",
    "make_spatial_mask",
    "make_spectral_mask",
    "persist_mask_config",
    # Tokenization + pos enc
    "SpectralTokenizer",
    "WavelengthPosEnc",
]
