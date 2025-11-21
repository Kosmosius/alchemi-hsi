from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .posenc import WavelengthPosEnc


@dataclass
class TokenizerConfig:
    context_size: int = 1
    num_pos_frequencies: int = 4
    include_log_lambda: bool = False
    include_normalized_wavelength: bool = False


class SpectralTokenizer(nn.Module):
    """Tokenize per-pixel spectra with optional spatial context and wavelength encodings."""

    def __init__(self, config: TokenizerConfig | None = None, **legacy_kwargs: object) -> None:
        super().__init__()
        if config is None:
            config = TokenizerConfig(**legacy_kwargs)
        if config.context_size not in (1, 3, 5):
            raise ValueError("context_size must be one of {1, 3, 5}")

        self.config = config
        self.pos_enc = WavelengthPosEnc(
            num_frequencies=self.config.num_pos_frequencies,
            include_log_lambda=self.config.include_log_lambda,
            include_normalized_wavelength=self.config.include_normalized_wavelength,
        )
        self.per_band_feature_dim = (
            self.config.context_size * self.config.context_size + self.pos_enc.output_dim
        )

    def forward(
        self,
        cube: torch.Tensor,
        wavelengths_nm: torch.Tensor,
        band_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | dict[str, object]]]:
        if cube.ndim == 3:
            cube = cube.unsqueeze(0)
        if cube.ndim != 4:
            raise ValueError("cube must have shape [B, H, W, Bands] or [H, W, Bands]")

        batch, height, width, bands = cube.shape

        w = wavelengths_nm
        if w.ndim == 1:
            w = w.unsqueeze(0)
        if w.ndim != 2:
            raise ValueError("wavelengths_nm must be 1-D or 2-D")

        if band_valid_mask is None:
            mask = torch.isfinite(w) & (w > 0)
        else:
            mask = band_valid_mask
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.ndim != 2:
                raise ValueError("band_valid_mask must be 1-D or 2-D")

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
        cube = cube * mask.view(batch, 1, 1, bands)

        pos_features = self.pos_enc(w.to(cube.device), mask)
        pos_dim = pos_features.shape[-1]

        pad = self.config.context_size // 2
        cube_chw = cube.permute(0, 3, 1, 2)

        if pad > 0:
            cube_chw = F.pad(cube_chw, (pad, pad, pad, pad), mode="replicate")

        patches = F.unfold(cube_chw, kernel_size=self.config.context_size)
        patches = patches.transpose(1, 2).reshape(
            batch, height * width, bands, self.config.context_size * self.config.context_size
        )
        patches = patches * mask.view(batch, 1, bands, 1)

        pos_expanded = pos_features.unsqueeze(1).expand(-1, height * width, -1, -1)
        band_features = torch.cat([patches, pos_expanded], dim=-1)
        tokens = band_features.reshape(batch, height * width, -1)

        attn_mask = mask.any(dim=1, keepdim=True).expand(batch, height * width)
        band_pad_mask = ~mask
        feature_mask = mask.view(batch, 1, bands).expand(batch, height * width, bands)
        if self.config.context_size > 1:
            feature_mask = feature_mask.repeat(
                1, 1, self.config.context_size * self.config.context_size
            )

        token_info: dict[str, torch.Tensor | dict[str, object]] = {
            "attn_mask": attn_mask,
            "band_pad_mask": band_pad_mask,
            "band_mask": feature_mask,
            "meta": {
                "context_size": self.config.context_size,
                "posenc_dim": pos_dim,
                "feature_dim_per_band": self.per_band_feature_dim,
                "spatial_shape": (height, width),
            },
        }

        return tokens, token_info


__all__ = ["SpectralTokenizer", "TokenizerConfig"]
