from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import torch
from torch import Tensor, nn

from ..masking import MaskingConfig, make_spatial_mask, make_spectral_mask


class MAEOutput(NamedTuple):
    recon: Tensor
    spatial_mask: Tensor  # shape: (num_tokens,)  True = keep
    spectral_mask: Tensor  # shape: (num_bands,)  True = keep
    loss: Tensor


class MAEEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 6,
        n_heads: int = 8,
        *,
        use_posenc: bool = True,
        posenc: nn.Module | None = None,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
        self.use_posenc = use_posenc
        # If positional encoding is enabled, either use the provided module or
        # fall back to a simple learnable embedding over token positions.
        self.posenc: nn.Module | None = (
            None if not use_posenc else (posenc or nn.Embedding(max_tokens, embed_dim))
        )

    def forward(self, tokens: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """Encode per-token embeddings.

        tokens: (B, T, D) or (T, D)  (we auto-add batch dim if needed)
        key_padding_mask: (B, T) or None
        """
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        if self.use_posenc and self.posenc is not None:
            positions = torch.arange(tokens.size(1), device=tokens.device)
            pos = self.posenc(positions)
            if pos.dim() == 2:
                pos = pos.unsqueeze(0)
            tokens = tokens + pos[:, : tokens.size(1), :]
        return self.enc(tokens, src_key_padding_mask=key_padding_mask)


class MAEDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 4,
        n_heads: int = 8,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            batch_first=True,
        )
        self.dec = nn.TransformerDecoder(layer, num_layers=depth)
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, z: Tensor, mem: Tensor | None = None) -> Tensor:
        """Decode to reconstruction tokens.

        z:   (B, T, D) queries
        mem: (B, T, D) memory (defaults to z)
        """
        if mem is None:
            mem = z
        y = self.dec(z, mem)
        return self.proj(y)


class MaskedAutoencoder(nn.Module):
    """Simple spatial+spectral masked autoencoder over per-pixel spectra.

    Expected input: tokens of shape (B, T, BANDS).
    """

    def __init__(
        self,
        embed_dim: int,
        out_dim: int,
        mask_cfg: MaskingConfig | None = None,
        depth: int = 6,
        n_heads: int = 8,
        decoder_depth: int | None = None,
    ) -> None:
        super().__init__()
        self.mask_cfg = mask_cfg or MaskingConfig()
        self.embed_dim = embed_dim

        self.encoder = MAEEncoder(embed_dim=embed_dim, depth=depth, n_heads=n_heads)
        self.decoder = MAEDecoder(
            embed_dim=embed_dim,
            depth=decoder_depth if decoder_depth is not None else max(1, depth // 2),
            n_heads=n_heads,
            out_dim=out_dim,
        )
        # Learned token used to fill masked spatial positions in the decoder memory.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(
        self,
        tokens: Tensor,
        *,
        persist_dir: str | Path | None = None,
        include_unmasked_loss: bool = True,
    ) -> MAEOutput:
        """Run spatial+spectral masking, encode, decode, and compute losses.

        tokens: (B, T, C) where T = tokens, C = bands.
        """
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        batch, num_tokens, bands = tokens.shape

        # --- build masks (True = keep, False = masked) ---
        spatial_mask = make_spatial_mask(
            num_tokens,
            self.mask_cfg.spatial_mask_ratio,
            self.mask_cfg.mask_seed,
        ).to(tokens.device)
        spectral_mask = make_spectral_mask(
            bands,
            self.mask_cfg.spectral_mask_ratio,
            self.mask_cfg.spectral_grouping,
            self.mask_cfg.mask_seed,
        ).to(tokens.device)

        # --- persist mask config if requested ---
        if persist_dir is not None:
            run_dir = Path(persist_dir)
            seed_str = (
                str(self.mask_cfg.mask_seed) if self.mask_cfg.mask_seed is not None else "default"
            )
            run_dir = run_dir / f"run-{seed_str}"
            self.mask_cfg.persist(run_dir)

        # --- apply spatial masking on tokens ---
        visible_tokens = tokens[:, spatial_mask]  # (B, T_keep, C)
        masked_tokens = tokens[:, ~spatial_mask]  # (B, T_mask, C)

        # Zero-out masked spectral bands on visible tokens.
        visible_tokens_masked = visible_tokens.clone()
        if visible_tokens_masked.numel() > 0:
            visible_tokens_masked[..., ~spectral_mask] = 0

        # --- encode only visible tokens ---
        if visible_tokens.shape[1] == 0:
            encoded = torch.zeros(
                batch,
                0,
                self.embed_dim,
                device=tokens.device,
                dtype=tokens.dtype,
            )
        else:
            encoded = self.encoder(visible_tokens_masked)

        # --- build full decoder memory: encoded visible + learned mask token ---
        full_memory = torch.empty(
            batch,
            num_tokens,
            encoded.shape[-1],
            device=encoded.device,
            dtype=encoded.dtype,
        )
        full_memory[:, spatial_mask] = encoded
        if masked_tokens.shape[1] > 0:
            full_memory[:, ~spatial_mask] = self.mask_token.expand(
                batch,
                masked_tokens.shape[1],
                -1,
            )

        # --- decode to reconstruct all tokens ---
        decoded = self.decoder(full_memory, mem=full_memory)

        # --- losses ---
        loss_terms: list[Tensor] = []

        # spatial loss: reconstruct masked tokens
        if (~spatial_mask).any():
            loss_spatial = torch.nn.functional.mse_loss(decoded[:, ~spatial_mask], masked_tokens)
            loss_terms.append(loss_spatial)

        # spectral loss: reconstruct masked bands on visible tokens
        spectral_weight = 1.0
        if (~spectral_mask).any() and spatial_mask.any():
            target_masked_bands = visible_tokens[..., ~spectral_mask]
            pred_masked_bands = decoded[:, spatial_mask][..., ~spectral_mask]
            loss_spectral = torch.nn.functional.mse_loss(pred_masked_bands, target_masked_bands)
            loss_terms.append(spectral_weight * loss_spectral)

        # optional global reconstruction loss (stability / regularisation)
        if include_unmasked_loss:
            loss_all = torch.nn.functional.mse_loss(decoded, tokens)
            loss_terms.append(0.1 * loss_all)

        total_loss = (
            torch.stack(loss_terms).sum() if loss_terms else torch.tensor(0.0, device=tokens.device)
        )

        return MAEOutput(decoded, spatial_mask, spectral_mask, total_loss)
