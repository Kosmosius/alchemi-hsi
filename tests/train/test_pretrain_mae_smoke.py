import math
from pathlib import Path

import yaml

from alchemi.training import run_pretrain_mae


def test_run_pretrain_mae_smoke(tmp_path, monkeypatch):
    config = {
        "global": {
            "device": "cpu",
            "dtype": "float32",
            "deterministic": True,
        },
        "train": {
            "max_steps": 2,
            "batch_size": 2,
            "embed_dim": 4,
            "n_heads": 2,
            "depth": 2,
            "basis_K": 4,
            "lr": 1e-3,
            "log_every": 1,
            "spatial_mask_ratio": 0.5,
            "spectral_mask_ratio": 0.5,
            "mask_path": str(tmp_path / "mask.pt"),
        },
    }

    config_path = Path(tmp_path) / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    monkeypatch.chdir(tmp_path)

    stats = run_pretrain_mae(str(config_path), seed_override=123)

    assert stats.tokens > 0
    assert math.isfinite(stats.tokens_per_s)
    assert stats.gb_per_s is None or math.isfinite(stats.gb_per_s)
