import math
from dataclasses import replace

from alchemi.train.alignment_trainer import AlignmentTrainer, load_alignment_config


def test_alignment_trainer_runs_smoke():
    cfg = load_alignment_config("configs/phase2/alignment.yaml")
    cfg.trainer = replace(cfg.trainer, max_steps=3, log_every=1, eval_every=2)

    trainer = AlignmentTrainer(cfg)
    history = trainer.train()

    assert len(history) == 3
    assert math.isfinite(history[-1]["loss"])
