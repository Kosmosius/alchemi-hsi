"""Configuration schemas for ALCHEMI experiments.

These Pydantic models mirror the configuration structure described in the
ALCHEMI design doc, covering dataset handling, model architecture, staged
training, evaluation, and top-level experiment wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from alchemi.utils.paths import resolve_path


class SpectrumConfig(BaseModel):
    """Basic spectral metadata for a sensor or dataset."""

    model_config = ConfigDict(extra="forbid")

    num_bands: int = Field(224, description="Number of spectral bands available")
    wavelength_min: float = Field(
        350.0,
        description="Minimum wavelength captured (nanometers)",
    )
    wavelength_max: float = Field(
        2500.0,
        description="Maximum wavelength captured (nanometers)",
    )
    resolution: float = Field(
        10.0,
        description="Approximate spectral resolution in nanometers",
    )


class SampleConfig(BaseModel):
    """Patch/chip sizing helpers used by :class:`DataConfig`."""

    model_config = ConfigDict(extra="forbid")

    chip_size: int = Field(256, description="Spatial chip size used when tiling scenes")
    patch_size: int = Field(32, description="Model ingest patch size")
    stride: int = Field(16, description="Stride between adjacent patches")


class AugmentationConfig(BaseModel):
    """Data augmentation switches aligned with the design doc data model section."""

    model_config = ConfigDict(extra="forbid")

    band_masking: bool = Field(
        True,
        description="Apply spectral band dropout/masking during training",
    )
    srf_jitter: bool = Field(
        True,
        description="Perturb spectral response functions to simulate sensor uncertainty",
    )
    noise_injection: bool = Field(
        False,
        description="Additive noise augmentation for robustness",
    )
    geometric_transforms: bool = Field(
        True,
        description="Spatial flips/rotations/crops enabled",
    )


class DataConfig(BaseModel):
    """Dataset- and sensor-related configuration.

    Captures dataset selection, sensor IDs, data splits, spatial sizing, and
    augmentation options described in the data model section of the design doc.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_names: list[str] = Field(
        default_factory=lambda: ["emit_solids"],
        description="List of dataset identifiers to load",
    )
    sensors: list[str] = Field(
        default_factory=lambda: ["emit"],
        description="Sensor identifiers used in this run",
    )
    splits: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "train": ["train"],
            "val": ["val"],
            "test": ["test"],
        },
        description="Mapping of logical splits to dataset split names",
    )
    sample: SampleConfig = Field(default_factory=SampleConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    data_roots: list[Path] = Field(
        default_factory=lambda: [Path("data")],
        description="Root directories for dataset storage",
    )
    spectrum: SpectrumConfig = Field(default_factory=SpectrumConfig)

    @field_validator("data_roots", mode="before")
    @classmethod
    def _coerce_roots(cls, value: Sequence[str | Path] | str | Path) -> list[Path]:
        roots: Sequence[str | Path]
        if isinstance(value, (str, Path)):
            roots = [value]
        else:
            roots = value
        return [resolve_path(root) for root in roots]


class IngestConfig(BaseModel):
    """Sensor-agnostic tokenization settings."""

    model_config = ConfigDict(extra="forbid")

    spectral_bins: int = Field(256, description="Number of spectral bins after resampling")
    patch_embed_dim: int = Field(256, description="Embedding dimension for patch tokens")
    max_sensors: int = Field(
        4,
        description="Maximum number of concurrent sensors in a batch",
    )
    dropout: float = Field(0.0, description="Dropout applied in the ingest stem")
    mask_tokens: bool = Field(
        True,
        description="Whether to include learned mask tokens for missing bands",
    )
    normalization: Literal["layernorm", "batchnorm"] = Field(
        "layernorm",
        description="Normalization applied to ingest outputs",
    )


class BackboneConfig(BaseModel):
    """MAE backbone configuration (size/depth/heads)."""

    model_config = ConfigDict(extra="forbid")

    variant: Literal["mae_base", "mae_large"] = "mae_base"
    embed_dim: int = Field(768, description="Backbone embedding dimension")
    depth: int = Field(12, description="Number of transformer blocks")
    num_heads: int = Field(12, description="Attention heads per block")
    grouping: int = Field(1, description="Spectral grouping factor")
    masking_ratio: float = Field(0.75, description="MAE masking ratio")
    decoder_dim: int = Field(512, description="Decoder embedding dimension")
    decoder_depth: int = Field(8, description="Number of decoder layers")


class AlignmentConfig(BaseModel):
    """Lab/overhead alignment losses and projections."""

    model_config = ConfigDict(extra="forbid")

    enable_contrastive: bool = True
    enable_cycle: bool = True
    contrastive_weight: float = 1.0
    cycle_weight: float = 1.0
    temperature: float = 0.07
    projection_dim: int = 256
    projection_depth: int = 2


class SolidsHeadConfig(BaseModel):
    """Solids classification/regression head settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    hidden_dim: int = 512
    dropout: float = 0.1
    loss_weight: float = 1.0


class GasHeadConfig(BaseModel):
    """Gas detection head settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    hidden_dim: int = 512
    dropout: float = 0.1
    loss_weight: float = 1.0


class AuxHeadConfig(BaseModel):
    """Auxiliary/representation head settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    projection_dim: int = 256
    loss_weight: float = 0.5


class HeadsConfig(BaseModel):
    """Container for all prediction heads."""

    model_config = ConfigDict(extra="forbid")

    solids: SolidsHeadConfig = Field(default_factory=SolidsHeadConfig)
    gas: GasHeadConfig = Field(default_factory=GasHeadConfig)
    aux: AuxHeadConfig = Field(default_factory=AuxHeadConfig)


class UncertaintyConfig(BaseModel):
    """Uncertainty estimation controls (Stage D)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    ensemble_size: int = 5
    conformal: bool = True
    conformal_alpha: float = 0.1
    temperature_scaling: bool = True


class ModelConfig(BaseModel):
    """Model architecture configuration spanning ingest/backbone/alignment/heads."""

    model_config = ConfigDict(extra="forbid")

    ingest: IngestConfig = Field(default_factory=IngestConfig)
    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    heads: HeadsConfig = Field(default_factory=HeadsConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)


class OptimizerConfig(BaseModel):
    """Optimizer hyperparameters (AdamW by default)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["adamw", "sgd"] = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    momentum: float | None = None


class SchedulerConfig(BaseModel):
    """Learning-rate schedule configuration."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["cosine", "constant"] = "cosine"
    warmup_steps: int = 1000
    min_lr: float = 1e-5
    max_epochs: int | None = None


class StageSetting(BaseModel):
    """Per-stage training controls (Stage A/B/C/D)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    epochs: int | None = None
    learning_rate: float | None = None
    loss_weight: float = 1.0
    description: str | None = None


class StageSchedule(BaseModel):
    """Staged training plan for MAE, alignment, task heads, and uncertainty."""

    model_config = ConfigDict(extra="forbid")

    mae: StageSetting = Field(
        default_factory=lambda: StageSetting(
            enabled=True,
            epochs=400,
            learning_rate=1.5e-4,
            description="Stage A: MAE spectral pretraining",
        ),
    )
    align: StageSetting = Field(
        default_factory=lambda: StageSetting(
            enabled=True,
            epochs=120,
            learning_rate=1e-4,
            description="Stage B: Lab/overhead alignment",
        ),
    )
    tasks: StageSetting = Field(
        default_factory=lambda: StageSetting(
            enabled=True,
            epochs=160,
            learning_rate=8e-5,
            description="Stage C: Task heads finetuning",
        ),
    )
    uncertainty: StageSetting = Field(
        default_factory=lambda: StageSetting(
            enabled=True,
            epochs=40,
            learning_rate=5e-5,
            description="Stage D: Uncertainty calibration",
        ),
    )


class MultiTaskConfig(BaseModel):
    """Multi-task loss balancing and gradient strategy toggles."""

    model_config = ConfigDict(extra="forbid")

    solids_weight: float = 1.0
    gas_weight: float = 1.0
    aux_weight: float = 0.5
    use_gradnorm: bool = False
    use_pcgrad: bool = False


class TrainingConfig(BaseModel):
    """Training hyperparameters across all stages and tasks."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = 64
    eval_batch_size: int = 64
    accumulation_steps: int = 1
    epochs: int = 200
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    stages: StageSchedule = Field(default_factory=StageSchedule)
    multitask: MultiTaskConfig = Field(default_factory=MultiTaskConfig)


class EvalConfig(BaseModel):
    """Evaluation suite selection per the design doc evaluation section."""

    model_config = ConfigDict(extra="forbid")

    solids: bool = True
    gas: bool = True
    representation: bool = True
    srf_robustness: bool = False
    heavy_atmosphere: bool = False
    teacher_noise: bool = False
    coverage: bool = True


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration that ties together all components."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    run_id: str | None = None
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
