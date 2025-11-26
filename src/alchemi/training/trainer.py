"""Synthetic MAE pretraining and toy alignment harness.

This module is primarily a lightweight throughput/ablation playground that
trains on random tokens. The CLIP-style alignment trainer in
``alchemi.train.alignment_trainer`` is the mainline path used for producing
deployable encoders. Keeping this module documented makes it clear that its
scope is experimental and synthetic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from ..config import RuntimeConfig
from ..heads import BandDepthHead, load_banddepth_config
from ..losses import InfoNCELoss, ReconstructionLoss, SpectralSmoothnessLoss
from ..models import (
    DomainDiscriminator,
    MAEDecoder,
    MAEEncoder,
    SetEncoder,
    SpectralBasisProjector,
    build_set_encoder,
)
from ..models import MaskingConfig
from ..utils.ckpt import save_checkpoint
from ..utils.logging import ThroughputMeter, ThroughputStats, get_logger
from .amp import autocast
from .config import TrainCfg
from .loss_mixer import Weights
from .seed import seed_everything

_LOG = get_logger(__name__)


def _is_main_process() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            return bool(torch.distributed.get_rank() == 0)
        except RuntimeError:
            pass

    for env_key in ("LOCAL_RANK", "RANK"):
        raw = os.environ.get(env_key)
        if raw is not None:
            try:
                return bool(int(raw) == 0)
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
    """Randomly mask a fraction of spectral positions."""
    B = values.shape[0]
    k = max(1, int(B * spectral_mask_ratio))
    idx = torch.randperm(B, device=values.device)[:k]
    m = mask.clone()
    m[idx] = False

    if persist_path is not None:
        persist_mask(m, persist_path)

    return m, idx


def _mask_spatial(shape: tuple[int, ...], spatial_mask_ratio: float, *, disabled: bool) -> Tensor:
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
    setenc = build_set_encoder(embed_dim=cfg.embed_dim, depth=2, heads=cfg.n_heads)
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


def _aggregate_stats(stats_list: list[ThroughputStats]) -> ThroughputStats:
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


def _normalise_train_config(
    config: str | Path | Mapping[str, Any] | TrainCfg,
    *,
    seed_override: int | None,
) -> tuple[RuntimeConfig, TrainCfg, str]:
    """Load a training configuration from various formats.

    Returns the runtime config, train config, and a human readable config label
    for logging/metadata.
    """
    config_label = "<in-memory-config>"
    if isinstance(config, TrainCfg):
        train_payload: Mapping[str, Any] = config.model_dump()
        payload: Mapping[str, Any] = {"train": train_payload}
    elif isinstance(config, Mapping):
        payload = config
        if "train" in payload:
            train_payload = payload.get("train", {}) or {}
        else:
            train_payload = payload
    else:
        config_path = Path(config)
        payload = yaml.safe_load(config_path.read_text())
        config_label = str(config_path)
        train_payload = payload.get("train", {}) if isinstance(payload, dict) else {}

    runtime_cfg = RuntimeConfig.from_mapping(
        payload.get("global") if isinstance(payload, Mapping) else None,
        fallback=train_payload,
    )
    runtime_cfg = runtime_cfg.with_seed(seed_override)
    cfg = TrainCfg(**train_payload) if not isinstance(config, TrainCfg) else config
    return runtime_cfg, cfg, config_label


def run_pretrain_mae(
    config: str | Path | Mapping[str, Any] | TrainCfg,
    *,
    no_spatial_mask: bool = False,
    no_posenc: bool = False,
    seed_override: int | None = None,
    return_loss: bool = False,
) -> ThroughputStats | tuple[ThroughputStats, float]:
    """Toy MAE pretraining loop with spatial+spectral masking and throughput measurement.

    Args:
        config: Path to a YAML config, a mapping containing ``global``/``train``
            sections, or an in-memory ``TrainCfg`` instance.
        no_spatial_mask: Disable spatial masking regardless of config value.
        no_posenc: Disable positional encoding regardless of config value.
        seed_override: Optional seed to override the config.
        return_loss: When ``True``, also return the final reconstruction loss
            from training (useful for ablations).

    This harness runs on synthetic tokens and is intended for ablations rather
    than producing the deployment encoder.
    """
    runtime_cfg, cfg, config_label = _normalise_train_config(
        config, seed_override=seed_override
    )

    # Fold CLI toggles into config so baselines are properly recorded.
    cfg = cfg.copy(
        update={
            "no_spatial_mask": cfg.no_spatial_mask or no_spatial_mask,
            "no_posenc": cfg.no_posenc or no_posenc,
        }
    )

    cfg = cfg.copy(update={"seed": runtime_cfg.seed, "deterministic": runtime_cfg.deterministic})

    # Seed / determinism config.
    torch.set_default_dtype(runtime_cfg.torch_dtype)
    seed_everything(runtime_cfg.seed, runtime_cfg.deterministic)
    device = runtime_cfg.torch_device
    _LOG.info(
        "Starting MAE pretrain config=%s seed=%s deterministic=%s device=%s dtype=%s",
        config_label,
        runtime_cfg.seed,
        runtime_cfg.deterministic,
        device,
        runtime_cfg.torch_dtype,
    )

    # Optional mask persistence path, if present in the config.
    mask_path: Path | None = None
    if hasattr(cfg, "mask_path"):
        mask_str = cfg.mask_path
        if mask_str:
            mask_path = Path(mask_str)

    basis, setenc = _build_embedder(cfg)

    no_spatial_mask = getattr(cfg, "no_spatial_mask", False)
    no_posenc = getattr(cfg, "no_posenc", False)
    mask_cfg = MaskingConfig(
        spatial_mask_ratio=cfg.spatial_mask_ratio,
        spectral_mask_ratio=cfg.spectral_mask_ratio,
    )

    enc = MAEEncoder(
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        n_heads=cfg.n_heads,
        use_posenc=not no_posenc,
        max_tokens=cfg.embed_dim * 2,
    )
    dec = MAEDecoder(
        embed_dim=cfg.embed_dim,
        depth=max(1, cfg.depth // 2),
        n_heads=cfg.n_heads,
        out_dim=cfg.embed_dim,
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

    meter = ThroughputMeter(device)
    step_stats: list[ThroughputStats] = []
    mask_persisted = False
    final_loss = 0.0

    # Simple synthetic MAE loop: random tokens with spatial+spectral masking.
    for step in range(1, cfg.max_steps + 1):
        batch_size = cfg.batch_size
        num_tokens = cfg.embed_dim
        embed_dim = cfg.embed_dim

        # Fake token grid: (B, T, D)
        tokens = torch.randn(
            batch_size, num_tokens, embed_dim, device=device, dtype=runtime_cfg.torch_dtype
        )

        # Spectral mask over feature dimension.
        spectral_values = torch.arange(
            embed_dim, dtype=runtime_cfg.torch_dtype, device=device
        )
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
        if persist_target is not None:
            mask_persisted = True

        # Spatial mask over tokens.
        spatial_mask = _mask_spatial(
            (num_tokens,),
            mask_cfg.spatial_mask_ratio,
            disabled=no_spatial_mask,
        ).to(device)

        # Encoder key padding mask: True = pad (i.e. masked-out tokens).
        key_padding_mask = (~spatial_mask).unsqueeze(0).expand(batch_size, -1)

        # Combined mask for reconstruction loss: (B, T, D)
        combined_mask_2d = spatial_mask.unsqueeze(-1) & spectral_mask.unsqueeze(0)
        combined_mask = combined_mask_2d.unsqueeze(0).expand(batch_size, -1, -1)

        # Throughput accounting.
        active_tokens = int(combined_mask.sum().item())
        element_size = tokens.element_size()
        num_bytes = int(tokens.numel() * element_size)

        meter.start()
        with autocast(enabled=False):
            z = enc(tokens, key_padding_mask=key_padding_mask)
            y = dec(z)

            # Reconstruction over masked region.
            loss = weights.recon * recon_loss(y, tokens, mask=combined_mask)

            # Spectral smoothness: first batch element with spectra in last dim
            values_for_smooth = tokens[0]  # (T, D)
            smooth_mask = spectral_mask.unsqueeze(0)
            smooth = smooth_loss(values_for_smooth, smooth_mask)
            loss = loss + weights.smooth * smooth

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        stats = meter.stop(tokens=active_tokens, num_bytes=num_bytes)
        step_stats.append(stats)
        final_loss = float(loss)

        if step % cfg.log_every == 0:
            _LOG.info(
                "[MAE] step %d loss=%.4f tokens/s=%.1f gb/s=%.2f peak_mem=%.2fGB",
                step,
                float(loss),
                stats.tokens_per_s,
                stats.gb_per_s or 0.0,
                stats.peak_mem_gb or 0.0,
            )

    save_checkpoint(
        "checkpoints/mae.pt",
        {"basis": basis.state_dict(), "enc": enc.state_dict(), "dec": dec.state_dict()},
    )

    aggregated = _aggregate_stats(step_stats)
    if return_loss:
        return aggregated, final_loss
    return aggregated


def run_align(config_path: str, *, seed_override: int | None = None) -> ThroughputStats:
    """Toy alignment loop with throughput measurement.

    This synthetic loop mirrors the main alignment trainer at a high level but
    runs on random tensors for fast instrumentation.
    """
    payload = yaml.safe_load(Path(config_path).read_text())
    train_payload = payload.get("train", {}) if isinstance(payload, dict) else {}
    runtime_cfg = RuntimeConfig.from_mapping(
        payload.get("global") if isinstance(payload, dict) else None,
        fallback=train_payload,
    )
    runtime_cfg = runtime_cfg.with_seed(seed_override)
    cfg = TrainCfg(**train_payload)

    cfg = cfg.copy(update={"seed": runtime_cfg.seed, "deterministic": runtime_cfg.deterministic})

    seed_everything(runtime_cfg.seed, runtime_cfg.deterministic)
    torch.set_default_dtype(runtime_cfg.torch_dtype)
    device = runtime_cfg.torch_device
    _LOG.info(
        "Starting ALIGN config=%s seed=%s deterministic=%s device=%s dtype=%s",
        config_path,
        runtime_cfg.seed,
        runtime_cfg.deterministic,
        device,
        runtime_cfg.torch_dtype,
    )

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

    # Synthetic paired feature/lab spectra
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
