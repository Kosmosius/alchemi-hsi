"""DDP-friendly training loop with precision toggles and checkpointing.

Intentionally small and self-contained so it can be used in tests and examples
without needing a full training stack.
"""
from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from spectra.utils.precision import PrecisionConfig, autocast
from spectra.utils.sdp import SDPBackend, select_sdp_backend


@dataclass
class TrainingConfig:
    """Configuration for the toy DDP training loop."""

    steps: int
    batch_size: int = 8
    input_dim: int = 16
    seed: int = 1234
    lr: float = 5e-3
    weight_decay: float = 0.0
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    sdp_preference: SDPBackend = "flash"
    gradient_checkpointing: bool = False
    checkpoint_path: Path | None = None


class ToyModel(nn.Module):
    """Tiny MLP with optional gradient checkpointing."""

    def __init__(self, input_dim: int, hidden_dim: int, gradient_checkpointing: bool) -> None:
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        def _forward(inp: torch.Tensor) -> torch.Tensor:
            act = torch.relu(self.proj(inp))
            normed = self.norm(act.float()).to(act.dtype)
            return self.head(normed).squeeze(-1)

        if self.gradient_checkpointing and self.training:
            return grad_checkpoint(_forward, x)
        return _forward(x)


def _seed_all(seed: int, rank: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs in a rank-dependent way."""
    torch.manual_seed(seed + rank)
    np.random.default_rng(seed + rank)
    random.seed(seed + rank)


@contextmanager
def distributed_setup(rank: int, world_size: int, port: int) -> Iterator[None]:
    """Initialize and tear down a process group for a local multi-process job."""
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_method = f"tcp://127.0.0.1:{port}"
    dist.init_process_group(backend, rank=rank, world_size=world_size, init_method=init_method)
    try:
        yield
    finally:
        dist.barrier()
        dist.destroy_process_group()


def _build_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


@dataclass
class CheckpointState:
    model: dict
    optimizer: dict
    scheduler: dict | None
    scaler: dict | None
    step: int
    config: dict


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: _LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler | None,
    step: int,
    cfg: TrainingConfig,
) -> None:
    """Persist a minimal training state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = CheckpointState(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict() if scheduler else None,
        scaler=scaler.state_dict() if scaler else None,
        step=step,
        config=asdict(cfg),
    )
    torch.save(state.__dict__, path)


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: _LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device | None = None,
) -> int:
    """Load model/optimizer/(optional) scheduler + scaler from disk.

    Returns the last completed step index.
    """
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    if device is not None:
        _move_optimizer_state(optimizer, device)
    if scheduler is not None and state_dict["scheduler"] is not None:
        scheduler.load_state_dict(state_dict["scheduler"])
    if scaler is not None and state_dict["scaler"] is not None:
        scaler.load_state_dict(state_dict["scaler"])
    return int(state_dict["step"])


def _generate_batch(
    step: int,
    rank: int,
    batch: int,
    input_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic synthetic regression batch used for tests.

    The generator is seeded from (step, rank) so baseline vs resumed runs see
    exactly the same inputs.
    """
    gen = torch.Generator(device=device).manual_seed(step * 997 + rank)
    inputs = torch.randn(batch, input_dim, generator=gen, device=device)
    target_w = torch.arange(1, input_dim + 1, device=device, dtype=torch.float32) / input_dim
    targets = torch.matmul(inputs, target_w) + 0.1
    return inputs, targets


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    cfg: TrainingConfig,
    rank: int,
    start_step: int = 0,
    device: torch.device | None = None,
) -> list[float]:
    """Run a full training run from start_step → cfg.steps on a single rank."""
    device = device or (
        torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    )
    if next(model.parameters()).device != device:
        model.to(device)
    ddp_model = DDP(model, device_ids=None if device.type == "cpu" else [rank])

    losses: list[float] = []

    for step in range(start_step, cfg.steps):
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = _generate_batch(step, rank, cfg.batch_size, cfg.input_dim, device)

        with select_sdp_backend(cfg.sdp_preference), autocast(cfg.precision):
            preds = ddp_model(inputs)
            loss = torch.nn.functional.mse_loss(preds.float(), targets.float())

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if rank == 0:
            losses.append(float(loss.detach().cpu()))

    return losses


def train_ddp(
    rank: int,
    world_size: int,
    port: int,
    cfg: TrainingConfig,
    resume: bool,
    log: list[float],
) -> None:
    """Entry point for mp.spawn.

    Each process:
      * seeds RNGs with a rank offset
      * joins a process group
      * optionally resumes from a checkpoint
      * runs the toy training loop
      * appends rank-0 losses into a shared list
    """
    _seed_all(cfg.seed, rank)

    with distributed_setup(rank, world_size, port):
        model = ToyModel(cfg.input_dim, cfg.input_dim * 2, cfg.gradient_checkpointing)
        optimizer = _build_optimizer(model, cfg)
        scaler = torch.cuda.amp.GradScaler(
            enabled=torch.cuda.is_available() and cfg.precision.resolved_precision() == "fp16"
        )

        device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        start_step = 0
        if resume and cfg.checkpoint_path is not None and cfg.checkpoint_path.exists():
            start_step = (
                load_checkpoint(cfg.checkpoint_path, model, optimizer, None, scaler, device) + 1
            )

        losses = train_one_epoch(model, optimizer, scaler, cfg, rank, start_step, device)

        if cfg.checkpoint_path is not None and rank == 0:
            save_checkpoint(cfg.checkpoint_path, model, optimizer, None, scaler, cfg.steps - 1, cfg)
            log.extend(losses)
        elif rank == 0:
            log.extend(losses)


__all__ = [
    "ToyModel",
    "TrainingConfig",
    "load_checkpoint",
    "save_checkpoint",
    "train_ddp",
]
