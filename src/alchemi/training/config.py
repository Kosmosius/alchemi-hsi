from pydantic import BaseModel, Field


class TrainCfg(BaseModel):
    mode: str = Field("mae", description="mae|align|joint")
    batch_size: int = 64
    lr: float = 3e-4
    max_steps: int = 10000
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 1000
    embed_dim: int = 256
    n_heads: int = 8
    depth: int = 6
    basis_K: int = 128


class DataCfg(BaseModel):
    sensors: list[str] = ["emit", "enmap", "avirisng", "hytes"]
    srf_root: str = "data/srf"
    paths: dict = {}
    wavelengths: dict = {}


class EvalCfg(BaseModel):
    sam_threshold: float = 0.1
    gas_fpr: float = 0.001
