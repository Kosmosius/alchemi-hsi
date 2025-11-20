from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from ..heads import BandDepthHead, load_banddepth_config
from ..losses import InfoNCELoss, ReconstructionLoss, SpectralSmoothnessLoss
from ..models import (
    DomainDiscriminator,
    MAEDecoder,
    MAEEncoder,
    SetEncoder,
    SpectralBasisProjector,
)
from ..masking import MaskingConfig
from ..utils.ckpt import save_checkpoint
from ..utils.logging import get_logger
from .seed import seed_everything
from .amp import autocast
from .config import TrainCfg
from .loss_mixer import Weights

_LOG = get_logger(__name__)


def _is_main_process() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            return torch.distributed.get_rank() == 0
        except RuntimeError:
            pass

    for env_key in ("LOCAL_RANK", "RANK"):
        raw = os.environ.get(env_key)
        if raw is not None:
            try:
                return int(raw) == 0
            except ValueError:
                continue

    return True


def persist_mask(mask: Tensor, path: Path) -> None:
    """Persist a boolean mask tensor for later inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mask.cpu(), path)


def _mask_spectral(
    values: Tensor,
    mask: Tensor,
    spectral_mask_ratio: float,
    *,
    persist_path: Path | None = None,
) -> tuple[Tensor, Tensor]:
    B = values.shape[0]
    k = max(1, int(B * spectral_mask_ratio))
    idx = torch.randperm(B)[:k]
    m = mask.clone()
    m[idx] = False

    if persist_path is not None:
        persist_mask(m, persist_path)

    return m, idx


def _mask_spatial(shape: Tuple[int, ...], spatial_mask_ratio: float, *, disabled: bool) -> Tensor:
    mask = torch.ones(shape, dtype=torch.bool)
    if disabled:
        return mask

    numel = mask.numel()
    k = max(1, int(numel * spatial_mask_ratio))
    flat = mask.view(-1)
    idx = torch.randperm(numel)[:k]
    flat[idx] = False
    return flat.view_as(mask)


def _build_embedder(cfg: TrainCfg) -> tuple[SpectralBasisProjector, SetEncoder]:
    basis = SpectralBasisProjector(K=cfg.basis_K)
    setenc = SetEncoder(dim=cfg.embed_dim, depth=2, heads=cfg.n_heads)
    return basis, setenc


def _encode_pixel(
    basis: SpectralBasisProjector,
    setenc: SetEncoder,
    wavelengths: Tensor,
    values: Tensor,
    mask: Tensor,
) -> Tensor:
    phi = basis(wavelengths, values, mask)
    tokens = phi.unsqueeze(0)
    return tokens.squeeze(0)


def run_pretrain_mae(
    config_path: str, *, no_spatial_mask: bool = False, no_posenc: bool = False
) -> None:
    cfg = TrainCfg(**yaml.safe_load(Path(config_path).read_text())["train"])

    # Fold CLI toggles into config so baselines are properly recorded.
    cfg = cfg.copy(
        update={
            "no_spatial_mask": cfg.no_spatial_mask or no_spatial_mask,
            "no_posenc": cfg.no_posenc or no_posenc,
        }
    )

    # Seed / determinism config.
    seed_everything(cfg.seed, cfg.deterministic)
    _LOG.info("Using seed=%s deterministic=%s", cfg.seed, cfg.deterministic)

    # Optional mask persistence path, if present in the config.
    mask_path: Path | None = None
    if hasattr(cfg, "mask_path"):
        mask_str = getattr(cfg, "mask_path")
        if mask_str:
            mask_path = Path(mask_str)

    basis, setenc = _build_embedder(cfg)

    mask_cfg = MaskingConfig(
        spatial_mask_ratio=cfg.spatial_mask_ratio,
        spectral_mask_ratio=cfg.spectral_mask_ratio,
        no_spatial_mask=cfg.no_spatial_mask,
        no_posenc=cfg.no_posenc,
    )

    enc = MAEEncoder(
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        n_heads=cfg.n_heads,
        use_posenc=not mask_cfg.no_posenc,
        max_tokens=cfg.embed_dim * 2,
    )
    dec = MAEDecoder(
        embed_dim=cfg.embed_dim,
        depth=max(1, cfg.depth // 2),
        n_heads=cfg.n_heads,
        out_dim=cfg.embed_dim,
    )

    recon_loss = ReconstructionLoss()
    smooth_loss = SpectralSmoothnessLoss()
    opt = torch.optim.AdamW(
        list(basis.parameters())
        + list(setenc.parameters())
        + list(enc.parameters())
        + list(dec.parameters()),
        lr=cfg.lr,
    )
    weights = Weights(recon=1.0, nce=0.0, sam=0.0, smooth=1e-4)

    mask_persisted = False
    device = next(enc.parameters()).device

    # Simple synthetic MAE loop: random tokens with spatial+spectral masking.
    for step in range(1, cfg.max_steps + 1):
        batch_size = cfg.batch_size
        num_tokens = cfg.embed_dim
        embed_dim = cfg.embed_dim

        # Fake token grid: (B, T, D)
        tokens = torch.randn(batch_size, num_tokens, embed_dim, device=device)

        # Spectral mask over feature dimension.
        spectral_values = torch.arange(embed_dim, dtype=torch.float32, device=device)
        base_band_mask = torch.ones(embed_dim, dtype=torch.bool, device=device)
        persist_target = (
            mask_path if (mask_path and not mask_persisted and _is_main_process()) else None
        )
        spectral_mask, _ = _mask_spectral(
            spectral_values,
            base_band_mask,
            spectral_mask_ratio=mask_cfg.spectral_mask_ratio,
            persist_path=persist_target,
        )
        mask_persisted = mask_persisted or persist_target is not None

        # Spatial mask over tokens.
        spatial_mask = _mask_spatial(
            (num_tokens,),
            mask_cfg.spatial_mask_ratio,
            disabled=mask_cfg.no_spatial_mask,
        ).to(device)

        # Encoder key padding mask: True = pad (i.e. masked-out tokens).
        key_padding_mask = (~spatial_mask).unsqueeze(0).expand(batch_size, -1)

        # Combined mask for reconstruction loss: (B, T, D)
        combined_mask_2d = spatial_mask.unsqueeze(-1) & spectral_mask.unsqueeze(0)
        combined_mask = combined_mask_2d.unsqueeze(0).expand(batch_size, -1, -1)

        with autocast(enabled=False):
            z = enc(tokens, key_padding_mask=key_padding_mask)
            y = dec(z)

            # Reconstruction over masked region.
            loss = weights.recon * recon_loss(y, tokens, mask=combined_mask)

            # Spectral smoothness: first batch element, shape (bands, tokens)
            values_for_smooth = tokens[0].transpose(0, 1)  # (D, T)
            smooth = smooth_loss(values_for_smooth, spectral_mask)
            loss = loss + weights.smooth * smooth

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg.log_every == 0:
            _LOG.info("[MAE] step %d loss=%.4f", step, float(loss))

    save_checkpoint(
        "checkpoints/mae.pt",
        {"basis": basis.state_dict(), "enc": enc.state_dict(), "dec": dec.state_dict()},
    )


def run_align(config_path: str) -> None:
    cfg = TrainCfg(**yaml.safe_load(Path(config_path).read_text())["train"])
    seed_everything(cfg.seed, cfg.deterministic)
    _LOG.info("Using seed=%s deterministic=%s", cfg.seed, cfg.deterministic)
    basis, setenc = _build_embedder(cfg)
    enc = MAEEncoder(embed_dim=cfg.embed_dim, depth=cfg.depth, n_heads=cfg.n_heads)
    nce = InfoNCELoss()
    domain = DomainDiscriminator(embed_dim=cfg.embed_dim, n_domains=4)

    params = (
        list(basis.parameters())
        + list(setenc.parameters())
        + list(enc.parameters())
        + list(domain.parameters())
    )
    band_head: BandDepthHead | None = None
    if cfg.banddepth_cfg and cfg.banddepth_weight > 0.0:
        bands = load_banddepth_config(cfg.banddepth_cfg)
        band_head = BandDepthHead(
            embed_dim=cfg.embed_dim,
            bands=bands,
            hidden_dim=cfg.banddepth_hidden,
            loss=cfg.banddepth_loss,
        )
        params += list(band_head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr)

    Xf = torch.randn(512, 64)
    Xl = torch.randn(512, 64)
    loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        list(zip(Xf, Xl, strict=False)), batch_size=cfg.batch_size, shuffle=True
    )

    for step, (f, lab) in enumerate(loader, start=1):
        with autocast(enabled=False):
            zf = enc(f.unsqueeze(1)).squeeze(1)
            zl = enc(lab.unsqueeze(1)).squeeze(1)
            loss = nce(zf, zl)
            if band_head is not None:
                band_head = band_head.to(zf.device)
                pooled = zf.mean(dim=1)
                preds = band_head(pooled)
                wavelengths = torch.linspace(
                    900.0,
                    2500.0,
                    f.shape[1],
                    device=f.device,
                    dtype=f.dtype,
                )
                targets = band_head.compute_targets(wavelengths, f.detach())
                band_loss = band_head.loss(preds, targets)
                loss = loss + cfg.banddepth_weight * band_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg.log_every == 0:
            _LOG.info("[ALIGN] step %d nce=%.4f", step, float(loss))
        if step >= cfg.max_steps:
            break
    save_checkpoint("checkpoints/align.pt", {"basis": basis.state_dict(), "enc": enc.state_dict()})


def run_eval(config_path: str) -> None:
    import numpy as np

    from ..eval.metrics_solids import macro_f1

    y_true = np.array([0, 1, 1, 0, 2])
    y_pred = np.array([0, 1, 0, 0, 2])
    f1 = macro_f1(y_true, y_pred)
    _LOG.info("macro-F1=%.3f", f1)
