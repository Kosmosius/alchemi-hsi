from __future__ import annotations

import socket
import tempfile
from multiprocessing import Manager
from pathlib import Path

import torch.multiprocessing as mp

from spectra.train.loop import TrainingConfig, train_ddp


def _spawn(world_size: int, cfg: TrainingConfig, resume: bool) -> list[float]:
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    manager = Manager()
    losses = manager.list()  # type: ignore[var-annotated]

    # Grab an ephemeral port for this run to avoid clashes across tests.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    mp.spawn(train_ddp, args=(world_size, port, cfg, resume, losses), nprocs=world_size, join=True)
    return list(losses)


def ddp_smoketest() -> None:
    world_size = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tempfile.NamedTemporaryFile(dir=tmpdir, delete=False).name)

        # Uninterrupted baseline.
        cfg = TrainingConfig(steps=20, checkpoint_path=None)
        baseline = _spawn(world_size, cfg, resume=False)

        # Two-phase run with checkpoint/resume.
        cfg_partial = TrainingConfig(steps=10, checkpoint_path=checkpoint_path)
        first_half = _spawn(world_size, cfg_partial, resume=False)

        resume_cfg = TrainingConfig(steps=20, checkpoint_path=checkpoint_path)
        second_half = _spawn(world_size, resume_cfg, resume=True)

    assert len(baseline) == 20
    assert len(first_half) == 10
    assert len(second_half) == 10

    resumed = first_half + second_half
    for idx, (ref, got) in enumerate(zip(baseline, resumed, strict=True)):
        if ref == 0:
            continue
        delta = abs(ref - got) / abs(ref)
        assert delta < 0.01, f"step {idx} diverged: ref={ref} got={got}"


def test_ddp_smoke() -> None:
    ddp_smoketest()


if __name__ == "__main__":
    ddp_smoketest()
