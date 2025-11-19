from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - distributed may be unavailable in test envs
    import torch.distributed as dist
except Exception:  # pragma: no cover - defensive
    dist = None

try:  # pragma: no cover - PyTorch < 1.10 fallback
    from torch.distributed.nn.functional import all_gather as _dist_all_gather
except Exception:  # pragma: no cover - defensive
    _dist_all_gather = None


@dataclass(slots=True)
class InfoNCELossOut:
    """Structured return type for the symmetric InfoNCE loss."""

    loss: torch.Tensor
    loss_lab_to_sensor: torch.Tensor
    loss_sensor_to_lab: torch.Tensor
    tau: torch.Tensor
    logit_scale: nn.Parameter | None = None

    def parameters(self) -> list[nn.Parameter]:
        if self.logit_scale is None:
            return []
        return [self.logit_scale]

    @property
    def metrics(self) -> dict[str, torch.Tensor]:
        return {
            "tau": self.tau.detach(),
            "loss_lab_to_sensor": self.loss_lab_to_sensor.detach(),
            "loss_sensor_to_lab": self.loss_sensor_to_lab.detach(),
        }


_TAU_PARAMS: dict[tuple[torch.device, torch.dtype], nn.Parameter] = {}


def _ddp_is_initialized() -> bool:
    return bool(
        dist and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    )


def _gather_embeddings(emb: torch.Tensor) -> torch.Tensor:
    """All-gather helper that is easy to stub in tests."""

    if not _ddp_is_initialized():
        return emb

    if _dist_all_gather is not None:
        gathered = _dist_all_gather(emb)
        if isinstance(gathered, torch.Tensor):
            return gathered.reshape(-1, emb.shape[-1])
        # Older PyTorch may return a list of tensors
        return torch.cat(list(gathered), dim=0)

    # Fallback to standard all_gather (non-differentiable but rarely hit)
    world_size = dist.get_world_size() if dist else 1
    chunks = [torch.zeros_like(emb) for _ in range(world_size)]
    if dist:
        dist.all_gather(chunks, emb)
    return torch.cat(chunks, dim=0)


def _get_tau_param(
    *,
    tau_init: float,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Parameter:
    key = (device, dtype)
    param = _TAU_PARAMS.get(key)
    if param is None:
        init_value = torch.log(torch.as_tensor(tau_init, device=device, dtype=dtype))
        param = nn.Parameter(init_value)
        _TAU_PARAMS[key] = param
    return param


def _cross_entropy_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive = logits.gather(1, target.unsqueeze(1)).squeeze(1)
    logsumexp = torch.logsumexp(logits, dim=1)
    return (logsumexp - positive).mean()


def info_nce_symmetric(
    z_lab: torch.Tensor,
    z_sensor: torch.Tensor,
    *,
    logit_scale: nn.Parameter | None = None,
    tau_init: float = 0.07,
    learnable_tau: bool = True,
    gather_ddp: bool = True,
) -> InfoNCELossOut:
    """CLIP-style symmetric InfoNCE with optional DDP gathering."""

    if z_lab.ndim != 2 or z_sensor.ndim != 2:
        msg = "Embeddings must be 2-D tensors [batch, dim]."
        raise ValueError(msg)
    if z_lab.shape != z_sensor.shape:
        msg = "z_lab and z_sensor must share shape."
        raise ValueError(msg)

    z_lab = F.normalize(z_lab, dim=-1)
    z_sensor = F.normalize(z_sensor, dim=-1)

    device = z_lab.device
    dtype = z_lab.dtype

    tau_param: nn.Parameter | None = None
    if logit_scale is not None:
        tau_param = logit_scale
        tau = torch.exp(logit_scale)
    elif learnable_tau:
        tau_param = _get_tau_param(tau_init=tau_init, device=device, dtype=dtype)
        tau = torch.exp(tau_param)
    else:
        tau = torch.as_tensor(tau_init, device=device, dtype=dtype)

    tau = torch.clamp(tau, min=1e-3, max=1e3)

    gather_active = gather_ddp and _ddp_is_initialized()
    if gather_active:
        world_size = dist.get_world_size() if dist else 1
        batch_size = z_lab.size(0)
        rank = dist.get_rank() if dist else 0
        gathered_lab = _gather_embeddings(z_lab)
        gathered_sensor = _gather_embeddings(z_sensor)
        positive_indices = torch.arange(batch_size, device=device) + rank * batch_size
    else:
        gathered_lab = z_lab
        gathered_sensor = z_sensor
        world_size = 1
        batch_size = z_lab.size(0)
        positive_indices = torch.arange(batch_size, device=device)

    logits_lab_to_sensor = z_lab @ gathered_sensor.T / tau
    logits_sensor_to_lab = z_sensor @ gathered_lab.T / tau

    loss_lab_to_sensor = _cross_entropy_from_logits(logits_lab_to_sensor, positive_indices)
    loss_sensor_to_lab = _cross_entropy_from_logits(logits_sensor_to_lab, positive_indices)
    loss = 0.5 * (loss_lab_to_sensor + loss_sensor_to_lab)

    return InfoNCELossOut(
        loss=loss,
        loss_lab_to_sensor=loss_lab_to_sensor,
        loss_sensor_to_lab=loss_sensor_to_lab,
        tau=tau.detach(),
        logit_scale=tau_param,
    )


# Backwards compatibility import path expected by older tests/configs.
LossOut = InfoNCELossOut


__all__ = ["InfoNCELossOut", "LossOut", "info_nce_symmetric"]
