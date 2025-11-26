import math
from pathlib import Path

import pytest
import yaml

from alchemi.training import run_pretrain_mae
from tests.io.test_ingest_golden import _write_emit_fixture

rasterio = pytest.importorskip("rasterio")


def test_run_pretrain_mae_real_data(tmp_path, monkeypatch):
    emit_path = tmp_path / "emit_fixture.tif"
    _write_emit_fixture(emit_path)

    config = {
        "global": {
            "device": "cpu",
            "dtype": "float32",
            "deterministic": True,
        },
        "data": {
            "mode": "real",
            "dataset_name": "emit_fixture",
            "paths": {"emit_fixture": str(emit_path)},
        },
        "train": {
            "max_steps": 2,
            "batch_size": 1,
            "embed_dim": 8,
            "n_heads": 2,
            "depth": 2,
            "basis_K": 8,
            "lr": 1e-3,
            "log_every": 1,
            "spatial_mask_ratio": 0.5,
            "spectral_mask_ratio": 0.5,
        },
    }

    config_path = Path(tmp_path) / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    monkeypatch.chdir(tmp_path)

    stats = run_pretrain_mae(str(config_path), seed_override=321)

    assert stats.tokens > 0
    assert math.isfinite(stats.tokens_per_s)
    assert stats.gb_per_s is None or math.isfinite(stats.gb_per_s)
