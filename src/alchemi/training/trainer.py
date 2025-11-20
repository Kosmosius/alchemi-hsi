from __future__ import annotations

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
from ..utils.ckpt import save_checkpoint
from ..utils.logging import get_logger
from .amp import autocast
from .config import TrainCfg
from .loss_mixer import Weights

_LOG = get_logger(__name__)


def _mask_spectral(
    values: Tensor, mask: Tensor, spectral_mask_ratio: float
) -> tuple[Tensor, Tensor]:
    B = values.shape[0]
    k = max(1, int(B * spectral_mask_ratio))
    idx = torch.randperm(B)[:k]
    m = mask.clone()
    m[idx] = False
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
    cfg = cfg.copy(
        update={
            "no_spatial_mask": cfg.no_spatial_mask or no_spatial_mask,
            "no_posenc": cfg.no_posenc or no_posenc,
        }
    )
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

    loader: DataLoader[Tensor] = DataLoader(
        TensorDataset(torch.randn(128, cfg.embed_dim)), batch_size=cfg.batch_size
    )

    step = 0
    for _batch in loader:
        step += 1
        wavelengths = torch.linspace(900.0, 2500.0, cfg.basis_K)
        values = torch.randn(cfg.basis_K, cfg.embed_dim // 4)
        spectral_mask, _ = _mask_spectral(
            wavelengths, torch.ones_like(wavelengths, dtype=torch.bool), spectral_mask_ratio=mask_cfg.spectral_mask_ratio
        )
        spatial_mask = _mask_spatial(values.shape, mask_cfg.spatial_mask_ratio, disabled=mask_cfg.no_spatial_mask)
        combined_mask = spatial_mask & spectral_mask.unsqueeze(-1)

        basis_tokens = basis(wavelengths, values.mean(dim=1), spectral_mask).unsqueeze(-1)
        spectral_tokens = torch.randn(cfg.basis_K, values.shape[1], cfg.embed_dim)
        spectral_tokens = spectral_tokens + values.unsqueeze(-1) + basis_tokens
        tokens_for_encoder = spectral_tokens.view(1, -1, cfg.embed_dim)
        context = setenc(spectral_tokens.view(-1, cfg.embed_dim), combined_mask.view(-1))
        tokens_for_encoder = tokens_for_encoder + context.view(1, 1, -1)
        key_padding_mask = ~combined_mask.view(1, -1)
        with autocast(enabled=False):
            z = enc(tokens_for_encoder, key_padding_mask=key_padding_mask)
            y = dec(z)
            target = spectral_tokens.view_as(y)
            loss = weights.recon * recon_loss(y, target, combined_mask.view(-1))
            loss = loss + weights.smooth * smooth_loss(values, combined_mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg.log_every == 0:
            _LOG.info(f"[MAE] step {step} loss={float(loss):.4f}")
        if step >= cfg.max_steps:
            break
    save_checkpoint(
        "checkpoints/mae.pt",
        {"basis": basis.state_dict(), "enc": enc.state_dict(), "dec": dec.state_dict()},
    )


def run_align(config_path: str) -> None:
    cfg = TrainCfg(**yaml.safe_load(Path(config_path).read_text())["train"])
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
            _LOG.info(f"[ALIGN] step {step} nce={float(loss):.4f}")
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
