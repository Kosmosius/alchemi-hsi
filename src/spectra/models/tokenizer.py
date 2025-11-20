from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .posenc import WavelengthPosEnc


class SpectralTokenizer(nn.Module):
    """Tokenize per-pixel spectra with optional spatial context and wavelength encodings.

    The tokenizer:
      * accepts hyperspectral cubes with shape ``[B, H, W, Bands]`` or ``[H, W, Bands]``,
      * handles variable band counts across samples via a boolean band-valid mask,
      * extracts per-band spatial patches (context windows) using ``F.unfold``,
      * concatenates per-band spatial features with wavelength positional encodings.

    The final token sequence has one token per spatial location (pixel); each token
    concatenates all band-wise features for that location.
    """

    def __init__(
        self,
        context_size: int = 1,
        num_pos_frequencies: int = 4,
        include_log_lambda: bool = False,
        include_normalized_wavelength: bool = False,
    ) -> None:
        """Args:
            context_size:
                Spatial context window size. Must be one of ``{1, 3, 5}``.
                A value of 1 means purely per-pixel spectra.
            num_pos_frequencies:
                Number of Fourier frequency bands used in the wavelength positional
                encoding.
            include_log_lambda:
                If True, include a normalized log-wavelength feature.
            include_normalized_wavelength:
                If True, include the normalized wavelength coordinate in the encoding.
        """
        super().__init__()
        if context_size not in (1, 3, 5):
            raise ValueError("context_size must be one of {1, 3, 5}")

        self.context_size = context_size
        self.pos_enc = WavelengthPosEnc(
            num_frequencies=num_pos_frequencies,
            include_log_lambda=include_log_lambda,
            include_normalized_wavelength=include_normalized_wavelength,
        )
        # For each band: all pixels in the context window + positional features.
        self.per_band_feature_dim = self.context_size * self.context_size + self.pos_enc.output_dim

    def forward(
        self,
        cube: torch.Tensor,
        wavelengths_nm: torch.Tensor,
        band_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor | Dict[str, object]]]:
        """Tokenize a hyperspectral cube.

        Args:
            cube:
                Spectral cube, shape ``[B, H, W, Bands]`` or ``[H, W, Bands]``.
            wavelengths_nm:
                Wavelengths in nanometers. Shape ``[B, Bands]`` or ``[Bands]``.
                If a single wavelength vector is given, it is broadcast across the
                batch dimension.
            band_valid_mask:
                Boolean mask indicating valid bands. Shape must match
                ``wavelengths_nm`` (up to the same broadcasting rules). If None,
                all bands are treated as valid.

        Returns:
            tokens:
                Tensor of shape ``[B, H*W, feature_dim]`` where
                ``feature_dim = Bands * per_band_feature_dim``.
            token_info:
                Dictionary with masks and metadata:
                  - ``attn_mask``: ``[B, H*W]`` bool mask over tokens (false if a
                    sample has no valid bands at all).
                  - ``band_pad_mask``: ``[B, Bands]`` bool mask where True marks
                    padded/invalid bands.
                  - ``meta``: additional information about shapes and configuration.
        """
        # Normalize cube to [B, H, W, C]
        if cube.ndim == 3:
            cube = cube.unsqueeze(0)
        if cube.ndim != 4:
            raise ValueError("cube must have shape [B, H, W, Bands] or [H, W, Bands]")

        batch, height, width, bands = cube.shape

        # Prepare wavelength grid and validity mask.
        w = wavelengths_nm
        if w.ndim == 1:
            w = w.unsqueeze(0)
        if w.ndim != 2:
            raise ValueError("wavelengths_nm must be 1-D or 2-D")

        if band_valid_mask is None:
            mask = torch.ones_like(w, dtype=torch.bool)
        else:
            mask = band_valid_mask
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.ndim != 2:
                raise ValueError("band_valid_mask must be 1-D or 2-D")

        # Broadcast wavelength grid / mask across batch if needed.
        if w.shape[0] == 1 and batch > 1:
            w = w.expand(batch, -1)
        if mask.shape[0] == 1 and batch > 1:
            mask = mask.expand(batch, -1)

        if w.shape[0] != batch:
            raise ValueError("Batch dimension of wavelengths_nm must match cube")
        if w.shape != mask.shape:
            raise ValueError("band_valid_mask must have the same shape as wavelengths_nm")
        if w.shape[1] != bands:
            raise ValueError("wavelength length must match cube spectral dimension")

        mask = mask.to(dtype=torch.bool, device=cube.device)
        cube = cube.to(device=cube.device)

        # Zero out padded bands in the spectra to avoid contaminating context windows.
        cube = cube * mask.view(batch, 1, 1, bands)

        # Positional encodings per band: [B, Bands, P]
        pos_features = self.pos_enc(w.to(cube.device), mask)
        pos_dim = pos_features.shape[-1]

        # Extract per-band spatial patches using unfold.
        pad = self.context_size // 2
        cube_chw = cube.permute(0, 3, 1, 2)  # [B, Bands, H, W]

        if pad > 0:
            cube_chw = F.pad(cube_chw, (pad, pad, pad, pad), mode="replicate")

        # F.unfold gives [B, Bands * K, H*W] where K = context_size**2
        patches = F.unfold(cube_chw, kernel_size=self.context_size)
        patches = patches.transpose(1, 2).reshape(
            batch, height * width, bands, self.context_size * self.context_size
        )
        patches = patches * mask.view(batch, 1, bands, 1)

        # Broadcast positional encodings over spatial locations: [B, H*W, Bands, P]
        pos_expanded = pos_features.unsqueeze(1).expand(-1, height * width, -1, -1)

        # Concatenate spatial patch and wavelength encodings per band.
        band_features = torch.cat([patches, pos_expanded], dim=-1)
        tokens = band_features.reshape(batch, height * width, -1)

        # Attention mask: tokens are valid if the sample has at least one valid band.
        attn_mask = mask.any(dim=1, keepdim=True).expand(batch, height * width)

        # Per-band padding mask: True means "this band is padded".
        band_pad_mask = ~mask

        token_info: Dict[str, torch.Tensor | Dict[str, object]] = {
            "attn_mask": attn_mask,
            "band_pad_mask": band_pad_mask,
            "meta": {
                "context_size": self.context_size,
                "posenc_dim": pos_dim,
                "feature_dim_per_band": self.per_band_feature_dim,
                "spatial_shape": (height, width),
            },
        }

        return tokens, token_info
