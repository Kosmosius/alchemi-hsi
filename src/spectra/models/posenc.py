from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class WavelengthPosEnc(nn.Module):
    """Fourier-style positional encodings for spectral wavelengths.

    Wavelengths are normalized to ``[0, 1]`` per sample over the valid bands and
    encoded with sine/cosine features at powers-of-two frequencies. Invalid bands
    are zeroed. Optionally, a normalized wavelength coordinate and a normalized
    log-wavelength channel are appended.

    Args:
        num_frequencies:
            Number of Fourier frequency bands. Each band contributes a sin/cos pair.
        include_log_lambda:
            If True, append a min-max normalized log-wavelength channel.
        include_normalized_wavelength:
            If True, append the normalized wavelength coordinate itself.
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

        # sin + cos per frequency, plus optional log and/or normalized coord
        self.output_dim = (
            2 * self.num_frequencies
            + (1 if self.include_log_lambda else 0)
            + (1 if self.include_normalized_wavelength else 0)
        )

    def forward(
        self,
        wavelengths_nm: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode wavelengths with mask-aware normalized Fourier features.

        Args:
            wavelengths_nm:
                Wavelengths in nanometers. Shape ``[B, Bands]`` or ``[Bands]``.
            valid_mask:
                Boolean mask of valid bands, same shape as ``wavelengths_nm`` (or
                1-D and broadcastable to it). If None, all bands are treated as valid.

        Returns:
            Tensor of shape ``[B, Bands, D]``, where ``D = output_dim``.
            Invalid bands are zeroed.
        """
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

        # Broadcast batch dimension if one of them is singleton.
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

        # Normalize wavelengths to [0, 1] per sample using only valid entries.
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
            # [B, Bands, 1]
            feature_chunks.append(norm.unsqueeze(-1))

        if self.num_frequencies > 0:
            # Powers-of-two frequency bands, scaled to [0, 2π · 2^(K-1)]
            freq_exponents = torch.arange(
                self.num_frequencies, device=w.device, dtype=w.dtype
            )
            freq_scales = (2.0 ** freq_exponents) * (2.0 * math.pi)
            angles = norm.unsqueeze(-1) * freq_scales  # [B, Bands, F]

            sin = torch.sin(angles)
            cos = torch.cos(angles)
            feature_chunks.extend((sin, cos))

        if self.include_log_lambda:
            # Min-max normalized log-wavelength, per sample over valid bands.
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
        # Extra safety: zero invalid bands in case of any numerical leakage.
        out = out * valid_mask.unsqueeze(-1)
        return out
