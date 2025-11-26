from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class TrainCfg(BaseModel):
    mode: str = Field("mae", description="mae|align|joint")
    batch_size: int = 64
    lr: float = 3e-4
    max_steps: int = 10000
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 1000
    seed: int = 42
    deterministic: bool = False
    mask_path: str | None = "checkpoints/mask.json"
    embed_dim: int = 256
    n_heads: int = 8
    depth: int = 6
    basis_K: int = 128
    spatial_mask_ratio: float = 0.75
    spectral_mask_ratio: float = 0.5
    no_spatial_mask: bool = False
    no_posenc: bool = False
    banddepth_cfg: str | None = None
    banddepth_weight: float = 0.0
    banddepth_loss: str = "l1"
    banddepth_hidden: int | None = None


class DataCfg(BaseModel):
    mode: Literal["synthetic", "real"] = "synthetic"
    dataset_name: Optional[str] = None
    sensors: list[str] = ["emit", "enmap", "avirisng", "hytes"]
    srf_root: str = "data/srf"
    paths: dict[str, Any] = {}
    wavelengths: dict[str, Any] = {}


class EvalCfg(BaseModel):
    sam_threshold: float = 0.1
    gas_fpr: float = 0.001
