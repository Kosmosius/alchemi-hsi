from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PosEncConfig:
    dim: int = 64
    max_freq_log2: int = 4


class WavelengthPositionalEncoding(nn.Module):
    """Fourier features computed over wavelengths expressed in nanometers."""

    def __init__(self, config: PosEncConfig | None = None) -> None:
        super().__init__()
        self.config = config or PosEncConfig()
        freq_bands = 2.0 ** torch.linspace(0, self.config.max_freq_log2, self.config.dim // 4)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def forward(self, wavelengths_nm: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        if wavelengths_nm.ndim != 2:
            raise ValueError("wavelengths_nm must be (batch, bands)")
        if pad_mask.shape != wavelengths_nm.shape:
            raise ValueError("pad_mask must match wavelengths")

        eps = 1e-6
        inf = torch.tensor(float("inf"), device=wavelengths_nm.device)
        valid_inf = torch.where(pad_mask, wavelengths_nm, inf)
        has_valid = valid_inf.isfinite().any(dim=1, keepdim=True)
        min_vals = valid_inf.min(dim=1, keepdim=True).values
        min_w = torch.where(has_valid, min_vals, torch.zeros_like(valid_inf[:, :1]))

        neg_inf = torch.tensor(float("-inf"), device=wavelengths_nm.device)
        valid_neg_inf = torch.where(pad_mask, wavelengths_nm, neg_inf)
        max_vals = valid_neg_inf.max(dim=1, keepdim=True).values
        max_w = torch.where(has_valid, max_vals, torch.ones_like(valid_neg_inf[:, :1]))

        norm = (wavelengths_nm - min_w) / (max_w - min_w + eps)
        norm = norm.clamp(0, 1)

        angles = norm.unsqueeze(-1) * self.freq_bands.to(norm.device) * 2 * math.pi
        sin_feat = torch.sin(angles)
        cos_feat = torch.cos(angles)
        enc = torch.cat([sin_feat, cos_feat], dim=-1)

        if enc.shape[-1] < self.config.dim:
            enc = nn.functional.pad(enc, (0, self.config.dim - enc.shape[-1]))
        elif enc.shape[-1] > self.config.dim:
            enc = enc[..., : self.config.dim]

        return enc * pad_mask.unsqueeze(-1)


class WavelengthPosEnc(nn.Module):
    """Fourier-style positional encodings for spectral wavelengths.

    This class remains backward compatible with the original API used in tests.
    """

    def __init__(
        self,
        num_frequencies: int = 4,
        include_log_lambda: bool = False,
        include_normalized_wavelength: bool = False,
    ) -> None:
        super().__init__()
        if num_frequencies < 0:
            raise ValueError("num_frequencies must be non-negative")

        self.num_frequencies = int(num_frequencies)
        self.include_log_lambda = bool(include_log_lambda)
        self.include_normalized_wavelength = bool(include_normalized_wavelength)

        self.output_dim = (
            2 * self.num_frequencies
            + (1 if self.include_log_lambda else 0)
            + (1 if self.include_normalized_wavelength else 0)
        )

    def forward(
        self,
        wavelengths_nm: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w = wavelengths_nm
        if w.ndim == 1:
            w = w.unsqueeze(0)
        if w.ndim != 2:
            raise ValueError("wavelengths_nm must be 1-D or 2-D")

        if valid_mask is None:
            valid_mask = torch.ones_like(w, dtype=torch.bool)
        else:
            m = valid_mask
            if m.ndim == 1:
                m = m.unsqueeze(0)
            if m.ndim != 2:
                raise ValueError("valid_mask must be 1-D or 2-D to match wavelengths_nm")
            valid_mask = m

        if w.shape[0] == 1 and valid_mask.shape[0] > 1:
            w = w.expand(valid_mask.shape[0], -1)
        if valid_mask.shape[0] == 1 and w.shape[0] > 1:
            valid_mask = valid_mask.expand(w.shape[0], -1)

        if w.shape != valid_mask.shape:
            raise ValueError("wavelengths_nm and valid_mask must have the same shape")

        if not w.is_floating_point():
            w = w.to(torch.get_default_dtype())
        valid_mask = valid_mask.to(dtype=torch.bool, device=w.device)

        eps = torch.finfo(w.dtype).eps
        mask_any = valid_mask.any(dim=1, keepdim=True)

        masked_min = torch.where(
            valid_mask, w, torch.full_like(w, float("inf"))
        ).amin(dim=1, keepdim=True)
        masked_max = torch.where(
            valid_mask, w, torch.full_like(w, float("-inf"))
        ).amax(dim=1, keepdim=True)

        default_min = torch.zeros_like(masked_min)
        default_max = torch.ones_like(masked_max)

        w_min = torch.where(mask_any, masked_min, default_min)
        w_max = torch.where(mask_any, masked_max, default_max)
        span = torch.clamp(w_max - w_min, min=eps)

        norm = (w - w_min) / span
        norm = torch.where(valid_mask, norm, torch.zeros_like(norm))

        feature_chunks = []

        if self.include_normalized_wavelength:
            feature_chunks.append(norm.unsqueeze(-1))

        if self.num_frequencies > 0:
            freq_exponents = torch.arange(
                self.num_frequencies, device=w.device, dtype=w.dtype
            )
            freq_scales = (2.0 ** freq_exponents) * (2.0 * math.pi)
            angles = norm.unsqueeze(-1) * freq_scales

            sin = torch.sin(angles)
            cos = torch.cos(angles)
            feature_chunks.extend((sin, cos))

        if self.include_log_lambda:
            log_w = torch.log(torch.clamp(w, min=eps))

            masked_log_min = torch.where(
                valid_mask, log_w, torch.full_like(log_w, float("inf"))
            ).amin(dim=1, keepdim=True)
            masked_log_max = torch.where(
                valid_mask, log_w, torch.full_like(log_w, float("-inf"))
            ).amax(dim=1, keepdim=True)

            default_log_min = torch.zeros_like(masked_log_min)
            default_log_max = torch.ones_like(masked_log_max)

            log_min = torch.where(mask_any, masked_log_min, default_log_min)
            log_max = torch.where(mask_any, masked_log_max, default_log_max)
            log_span = torch.clamp(log_max - log_min, min=eps)

            log_norm = (log_w - log_min) / log_span
            log_norm = torch.where(valid_mask, log_norm, torch.zeros_like(log_norm))

            feature_chunks.append(log_norm.unsqueeze(-1))

        if not feature_chunks:
            return torch.zeros((*w.shape, 0), device=w.device, dtype=w.dtype)

        out = torch.cat(feature_chunks, dim=-1)
        out = out * valid_mask.unsqueeze(-1)
        return out


__all__ = ["PosEncConfig", "WavelengthPosEnc", "WavelengthPositionalEncoding"]
