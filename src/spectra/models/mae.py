from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from spectra.models.masking import MaskingConfig, MaskingHelper
from spectra.models.posenc import PosEncConfig, WavelengthPositionalEncoding
from spectra.models.tokenizer import SpectralTokenizer, TokenizerConfig
from spectra.utils.precision import autocast_from_config
from spectra.utils.sdp import select_sdp_backend


@dataclass
class MAEConfig:
    encoder_dim: int = 256
    encoder_layers: int = 4
    decoder_dim: int = 128
    decoder_layers: int = 2
    num_heads: int = 4
    context_size: int = 1
    mask_ratio_spatial: float = 0.5
    mask_ratio_spectral: float = 0.5
    sam_loss: bool = False
    flash_attn: bool = False
    precision: str = "bf16"


class SpectralMAE(nn.Module):
    def __init__(
        self,
        config: Optional[MAEConfig] = None,
        posenc_config: Optional[PosEncConfig] = None,
        tokenizer_config: Optional[TokenizerConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config or MAEConfig()
        self.tokenizer = SpectralTokenizer(tokenizer_config or TokenizerConfig(context_size=self.config.context_size))
        self.posenc = WavelengthPositionalEncoding(posenc_config or PosEncConfig(dim=self.config.encoder_dim))
        self.masker = MaskingHelper(
            MaskingConfig(
                spatial_mask_ratio=self.config.mask_ratio_spatial,
                spectral_mask_ratio=self.config.mask_ratio_spectral,
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.encoder_dim,
            nhead=self.config.num_heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.encoder_layers)
        self.proj_in = nn.Linear(1, self.config.encoder_dim)
        self.pos_projection = nn.Linear(self.config.encoder_dim, self.config.encoder_dim)
        self.encoder_input = nn.Linear(self.config.encoder_dim, self.config.encoder_dim, bias=True)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.config.encoder_dim,
                nhead=max(1, self.config.num_heads // 2),
                batch_first=True,
            ),
            num_layers=self.config.decoder_layers,
        )
        self.band_query_proj = nn.Linear(self.config.encoder_dim, self.config.encoder_dim)
        self.reconstruction_head = nn.Linear(self.config.encoder_dim, 1)

    def forward(self, x: torch.Tensor, wavelengths_nm: torch.Tensor) -> dict:
        tokens, info = self.tokenizer(x, wavelengths_nm)
        band_mask = ~info["band_pad_mask"]  # type: ignore[index]
        attn_mask = info["attn_mask"].to(dtype=torch.bool)  # type: ignore[index]

        # Reshape raw tokens into [B, tokens, bands, per_band_feature_dim]
        context_area = self.tokenizer.config.context_size * self.tokenizer.config.context_size
        batch, num_tokens, _ = tokens.shape
        bands = band_mask.shape[1]
        per_band_dim = int(info["meta"]["feature_dim_per_band"])  # type: ignore[index]
        tokens_view = tokens.view(batch, num_tokens, bands, per_band_dim)

        pos_enc = self.posenc(wavelengths_nm, band_mask)
        pos_tokens = pos_enc.unsqueeze(1)

        # Embed spectral values (mean over the spatial context) and add wavelength encoding.
        spectral_features = tokens_view[..., :context_area]
        spec_values = spectral_features.mean(dim=-1, keepdim=True)
        tokens_emb = self.proj_in(spec_values) + self.pos_projection(pos_tokens)

        # Build masks for spatial and spectral dropout.
        masking = self.masker.combined_mask(tokens_emb.view(batch, num_tokens, -1), band_mask)
        spatial_mask = masking["spatial_mask"]
        spectral_mask = masking["spectral_mask"]

        # Aggregate bands into per-token summaries to keep encoder input shape fixed.
        band_mask_expanded = band_mask.unsqueeze(1).unsqueeze(-1)
        band_counts = band_mask_expanded.sum(dim=2).clamp(min=1)
        token_summary = (tokens_emb * band_mask_expanded).sum(dim=2) / band_counts

        visible = token_summary.masked_fill(spatial_mask.unsqueeze(-1), 0.0)
        key_padding_mask = ~attn_mask
        with select_sdp_backend("flash" if self.config.flash_attn else "mem_efficient"):
            with autocast_from_config({"precision": self.config.precision}):
                encoded = self.encoder(
                    self.encoder_input(visible),
                    src_key_padding_mask=key_padding_mask,
                )
        decoded = self.decoder(
            encoded,
            memory=encoded,
            tgt_key_padding_mask=key_padding_mask,
            memory_key_padding_mask=key_padding_mask,
        )

        # Condition reconstruction on per-band positional embeddings.
        band_queries = self.band_query_proj(pos_tokens).expand(-1, num_tokens, -1, -1)
        decoded_expanded = decoded.unsqueeze(2).expand(-1, -1, bands, -1)
        recon_spec = self.reconstruction_head(decoded_expanded + band_queries).squeeze(-1)

        target = spec_values.squeeze(-1)
        spectral_mask_expanded = spectral_mask.unsqueeze(1).expand_as(recon_spec)
        spatial_mask_expanded = spatial_mask.unsqueeze(-1).expand_as(recon_spec)
        total_mask = (spectral_mask_expanded | spatial_mask_expanded) & band_mask.unsqueeze(1)
        loss = (
            F.mse_loss(recon_spec[total_mask], target[total_mask])
            if total_mask.any()
            else torch.tensor(0.0, device=x.device)
        )

        return {
            "loss": loss,
            "reconstruction": recon_spec,
            "spatial_mask": spatial_mask,
            "spectral_mask": spectral_mask,
            "attn_mask": attn_mask,
            "band_mask": band_mask,
        }


__all__ = ["MAEConfig", "SpectralMAE"]
