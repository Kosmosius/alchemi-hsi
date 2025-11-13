from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

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


def _mask_spectral(values, mask, spectral_mask_ratio: float):
    B = values.shape[0]
    k = max(1, int(B * spectral_mask_ratio))
    idx = torch.randperm(B)[:k]
    m = mask.clone()
    m[idx] = False
    return m, idx


def _build_embedder(cfg: TrainCfg):
    basis = SpectralBasisProjector(K=cfg.basis_K)
    setenc = SetEncoder(dim=cfg.embed_dim, depth=2, heads=cfg.n_heads)
    return basis, setenc


def _encode_pixel(basis, setenc, wavelengths, values, mask):
    phi = basis(wavelengths, values, mask)
    tokens = phi.unsqueeze(0)
    return tokens.squeeze(0)


def run_pretrain_mae(config_path: str):
    cfg = TrainCfg(**yaml.safe_load(Path(config_path).read_text())["train"])
    basis, setenc = _build_embedder(cfg)
    enc = MAEEncoder(embed_dim=cfg.embed_dim, depth=cfg.depth, n_heads=cfg.n_heads)
    dec = MAEDecoder(
        embed_dim=cfg.embed_dim, depth=max(1, cfg.depth // 2), n_heads=cfg.n_heads, out_dim=1
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

    loader = DataLoader(TensorDataset(torch.randn(128, 32)), batch_size=cfg.batch_size)

    step = 0
    for _batch in loader:
        step += 1
        x = torch.randn(64, 64)
        band_mask = torch.ones(64, dtype=torch.bool)
        masked, _ = _mask_spectral(x[:, 0], band_mask, spectral_mask_ratio=0.5)
        with autocast(enabled=False):
            z = enc(x.unsqueeze(0))
            y = dec(z)
            loss = weights.recon * recon_loss(y.squeeze(0), x, masked)
            loss = loss + weights.smooth * smooth_loss(x, masked)
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


def run_align(config_path: str):
    cfg = TrainCfg(**yaml.safe_load(Path(config_path).read_text())["train"])
    basis, setenc = _build_embedder(cfg)
    enc = MAEEncoder(embed_dim=cfg.embed_dim, depth=cfg.depth, n_heads=cfg.n_heads)
    nce = InfoNCELoss()
    domain = DomainDiscriminator(embed_dim=cfg.embed_dim, n_domains=4)
    opt = torch.optim.AdamW(
        list(basis.parameters())
        + list(setenc.parameters())
        + list(enc.parameters())
        + list(domain.parameters()),
        lr=cfg.lr,
    )

    Xf = torch.randn(512, 64)
    Xl = torch.randn(512, 64)
    loader = DataLoader(list(zip(Xf, Xl, strict=False)), batch_size=cfg.batch_size, shuffle=True)

    for step, (f, lab) in enumerate(loader, start=1):
        with autocast(enabled=False):
            zf = enc(f.unsqueeze(1)).squeeze(1)
            zl = enc(lab.unsqueeze(1)).squeeze(1)
            loss = nce(zf, zl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg.log_every == 0:
            _LOG.info(f"[ALIGN] step {step} nce={float(loss):.4f}")
        if step >= cfg.max_steps:
            break
    save_checkpoint("checkpoints/align.pt", {"basis": basis.state_dict(), "enc": enc.state_dict()})


def run_eval(config_path: str):
    import numpy as np

    from ..eval.metrics_solids import macro_f1

    y_true = np.array([0, 1, 1, 0, 2])
    y_pred = np.array([0, 1, 0, 0, 2])
    f1 = macro_f1(y_true, y_pred)
    _LOG.info("macro-F1=%.3f", f1)
