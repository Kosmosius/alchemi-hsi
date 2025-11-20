from __future__ import annotations

from pathlib import Path

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
from ..utils.logging import ThroughputMeter, ThroughputStats, get_logger
from .amp import autocast
from .config import TrainCfg
from .loss_mixer import Weights

_LOG = get_logger(__name__)


def _mask_spectral(
    values: Tensor, mask: Tensor, spectral_mask_ratio: float
) -> tuple[Tensor, Tensor]:
    """Randomly mask a fraction of spectral positions."""
    B = values.shape[0]
    k = max(1, int(B * spectral_mask_ratio))
    idx = torch.randperm(B, device=values.device)[:k]
    m = mask.clone()
    m[idx] = False
    return m, idx


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


def _aggregate_stats(
    stats_list: list[ThroughputStats],
) -> ThroughputStats:
    """Aggregate per-step stats into a single summary."""
    if not stats_list:
        return ThroughputStats(
            tokens_per_s=0.0,
            gb_per_s=0.0,
            peak_mem_gb=None,
            step_time_s=0.0,
            tokens=0,
            num_bytes=0,
        )

    total_tokens = sum(int(s.tokens) for s in stats_list)
    total_bytes = sum(int(s.num_bytes or 0) for s in stats_list)
    total_time = sum(float(s.step_time_s) for s in stats_list)
    peak_mem = max(float(s.peak_mem_gb or 0.0) for s in stats_list)

    total_time = max(total_time, 1e-12)
    tokens_per_s = total_tokens / total_time
    gb_per_s = total_bytes / (total_time * 1e9)
    avg_step_time = total_time / len(stats_list)

    return ThroughputStats(
        tokens_per_s=tokens_per_s,
        gb_per_s=gb_per_s,
        peak_mem_gb=peak_mem if peak_mem > 0.0 else None,
        step_time_s=avg_step_time,
        tokens=total_tokens,
        num_bytes=total_bytes,
    )


def run_pretrain_mae(config_path: str) -> ThroughputStats:
    """Toy MAE pretraining loop with throughput measurement."""
    cfg = TrainCfg(**yaml.safe_load(Path(config_path).read_text())["train"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    basis, setenc = _build_embedder(cfg)
    enc = MAEEncoder(embed_dim=cfg.embed_dim, depth=cfg.depth, n_heads=cfg.n_heads)
    dec = MAEDecoder(
        embed_dim=cfg.embed_dim, depth=max(1, cfg.depth // 2), n_heads=cfg.n_heads, out_dim=1
    )

    for module in (basis, setenc, enc, dec):
        module.to(device)

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
        TensorDataset(torch.randn(128, 32)), batch_size=cfg.batch_size
    )

    meter = ThroughputMeter(device)
    step_stats: list[ThroughputStats] = []

    for step, _batch in enumerate(loader, start=1):
        x = torch.randn(64, 64, device=device)
        band_mask = torch.ones(64, dtype=torch.bool, device=device)
        masked, _ = _mask_spectral(x[:, 0], band_mask, spectral_mask_ratio=0.5)

        tokens = int(x.numel())
        num_bytes = int(tokens * x.element_size())

        meter.start()
        with autocast(enabled=False):
            z = enc(x.unsqueeze(0))
            y = dec(z)
            loss = weights.recon * recon_loss(y.squeeze(0), x, masked)
            loss = loss + weights.smooth * smooth_loss(x, masked)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        stats = meter.stop(tokens=tokens, num_bytes=num_bytes)
        step_stats.append(stats)

        if step % cfg.log_every == 0:
            _LOG.info(
                "[MAE] step %d loss=%.4f tokens/s=%.1f gb/s=%.2f peak_mem=%.2fGB",
                step,
                float(loss),
                stats.tokens_per_s,
                stats.gb_per_s or 0.0,
                stats.peak_mem_gb or 0.0,
            )

        if step >= cfg.max_steps:
            break

    save_checkpoint(
        "checkpoints/mae.pt",
        {"basis": basis.state_dict(), "enc": enc.state_dict(), "dec": dec.state_dict()},
    )

    return _aggregate_stats(step_stats)


def run_align(config_path: str) -> ThroughputStats:
    """Toy alignment loop with throughput measurement."""
    cfg = TrainCfg(**yaml.safe_load(Path(config_path).read_text())["train"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    basis, setenc = _build_embedder(cfg)
    enc = MAEEncoder(embed_dim=cfg.embed_dim, depth=cfg.depth, n_heads=cfg.n_heads)
    nce = InfoNCELoss()
    domain = DomainDiscriminator(embed_dim=cfg.embed_dim, n_domains=4)

    for module in (basis, setenc, enc, domain):
        module.to(device)

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

    meter = ThroughputMeter(device)
    step_stats: list[ThroughputStats] = []

    for step, (f, lab) in enumerate(loader, start=1):
        f = f.to(device)
        lab = lab.to(device)

        tokens = int(f.numel() + lab.numel())
        num_bytes = int(tokens * f.element_size())

        meter.start()
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
        stats = meter.stop(tokens=tokens, num_bytes=num_bytes)
        step_stats.append(stats)

        if step % cfg.log_every == 0:
            _LOG.info(
                "[ALIGN] step %d nce=%.4f tokens/s=%.1f gb/s=%.2f peak_mem=%.2fGB",
                step,
                float(loss),
                stats.tokens_per_s,
                stats.gb_per_s or 0.0,
                stats.peak_mem_gb or 0.0,
            )

        if step >= cfg.max_steps:
            break

    save_checkpoint("checkpoints/align.pt", {"basis": basis.state_dict(), "enc": enc.state_dict()})
    return _aggregate_stats(step_stats)


def run_eval(config_path: str) -> None:
    """Placeholder eval that wires up metric plumbing."""
    import numpy as np

    from ..eval.metrics_solids import macro_f1

    y_true = np.array([0, 1, 1, 0, 2])
    y_pred = np.array([0, 1, 0, 0, 2])
    f1 = macro_f1(y_true, y_pred)
    _LOG.info("macro-F1=%.3f", f1)
