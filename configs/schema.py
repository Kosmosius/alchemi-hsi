"""Schema helpers for experiment configuration files."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GlobalConfig(BaseModel):  # type: ignore[misc]
    seed: int = Field(42, description="Random seed applied across libraries")
    device: str = Field(
        "auto", description="Torch device string; 'auto' picks CUDA if available"
    )
    dtype: str = Field("float32", description="Default floating point dtype")
    amp_dtype: str | None = Field(
        None, description="Autocast dtype (bf16/fp16) when AMP is enabled"
    )
    deterministic: bool = Field(
        False, description="Enable deterministic algorithms when supported"
    )


class Config(BaseModel):  # type: ignore[misc]
    project: str | None = None
    global_: GlobalConfig = Field(default_factory=GlobalConfig, alias="global")

    class Config:
        allow_population_by_field_name = True
