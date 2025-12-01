"""Hydra-style configuration loader for ALCHEMI experiments.

This module composes YAML files under ``configs/`` into typed
:class:`~alchemi.config.core.ExperimentConfig` objects. Hydra is not a required
runtime dependency; if it is unavailable we simply rely on PyYAML and a small
amount of manual composition inspired by Hydra defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from alchemi.utils.paths import find_project_root

from .core import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    StageSetting,
    TrainingConfig,
)

CONFIG_ROOT = find_project_root() / "configs"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected mapping in {path}, found {type(data)}")
    return dict(data)


def _resolve_named_config(section: str, name_or_mapping: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(name_or_mapping, str):
        return _load_yaml(CONFIG_ROOT / section / f"{name_or_mapping}.yaml")
    if isinstance(name_or_mapping, Mapping):
        return dict(name_or_mapping)
    raise TypeError(f"Unsupported config reference for {section}: {name_or_mapping!r}")


def _resolve_model(model_raw: Any) -> dict[str, Any]:
    resolved = _resolve_named_config("model", model_raw)
    # Allow nested references for model sub-components.
    ingest = resolved.get("ingest")
    if isinstance(ingest, (str, Mapping)):
        resolved["ingest"] = _resolve_named_config("model/ingest", ingest)
    backbone = resolved.get("backbone")
    if isinstance(backbone, (str, Mapping)):
        resolved["backbone"] = _resolve_named_config("model/backbone", backbone)
    alignment = resolved.get("alignment")
    if isinstance(alignment, (str, Mapping)):
        resolved["alignment"] = _resolve_named_config("model/alignment", alignment)
    heads = resolved.get("heads")
    if isinstance(heads, Mapping):
        updated_heads: dict[str, Any] = {}
        for head_name, value in heads.items():
            if isinstance(value, (str, Mapping)):
                updated_heads[head_name] = _resolve_named_config("model/heads", value)
            else:
                updated_heads[head_name] = value
        resolved["heads"] = updated_heads
    uncertainty = resolved.get("uncertainty")
    if isinstance(uncertainty, (str, Mapping)):
        resolved["uncertainty"] = _resolve_named_config("model/uncertainty", uncertainty)
    return resolved


def _resolve_training(training_raw: Any) -> dict[str, Any]:
    resolved = _resolve_named_config("train", training_raw)
    stages = resolved.get("stages")
    if isinstance(stages, Mapping):
        updated_stages: dict[str, Any] = {}
        for stage_name, value in stages.items():
            if isinstance(value, (str, Mapping)):
                updated_stages[stage_name] = _resolve_named_config("train", value)
            else:
                updated_stages[stage_name] = value
        resolved["stages"] = updated_stages
    multitask = resolved.get("multitask")
    if isinstance(multitask, (str, Mapping)):
        resolved["multitask"] = _resolve_named_config("train", multitask)
    return resolved


def _resolve_eval(eval_raw: Any) -> dict[str, Any]:
    return _resolve_named_config("eval", eval_raw)


def _resolve_data(data_raw: Any) -> dict[str, Any]:
    return _resolve_named_config("data", data_raw)


def load_data_config(name_or_mapping: str | Mapping[str, Any]) -> DataConfig:
    """Load a :class:`DataConfig` from ``configs/data`` or a mapping."""

    return DataConfig.model_validate(_resolve_data(name_or_mapping))


def load_model_config(name_or_mapping: str | Mapping[str, Any]) -> ModelConfig:
    """Load a :class:`ModelConfig` from model sub-configs or a mapping."""

    return ModelConfig.model_validate(_resolve_model(name_or_mapping))


def load_training_config(name_or_mapping: str | Mapping[str, Any]) -> TrainingConfig:
    """Load a :class:`TrainingConfig` with staged training references resolved."""

    raw_training = _resolve_training(name_or_mapping)
    stages = raw_training.get("stages")
    if isinstance(stages, Mapping):
        validated_stages: dict[str, StageSetting] = {}
        for stage_name, stage_value in stages.items():
            validated_stages[stage_name] = StageSetting.model_validate(stage_value)
        raw_training["stages"] = validated_stages
    return TrainingConfig.model_validate(raw_training)


def load_eval_config(name_or_mapping: str | Mapping[str, Any]) -> EvalConfig:
    """Load an :class:`EvalConfig`."""

    return EvalConfig.model_validate(_resolve_eval(name_or_mapping))


def load_experiment_config(name: str | Path, config_root: str | Path | None = None) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` for a named experiment.

    Parameters
    ----------
    name:
        Either the name of an experiment file (without ``.yaml``) under
        ``configs/experiment`` or a direct path to a YAML file.
    config_root:
        Optional override for the configs directory. Useful for tests.
    """

    root = Path(config_root) if config_root is not None else CONFIG_ROOT
    path = Path(name)
    if not path.suffix:
        path = root / "experiment" / f"{path}.yaml"
    elif not path.is_absolute():
        path = root / path

    raw = _load_yaml(path)

    data_cfg = load_data_config(raw.get("data", {}))
    model_cfg = load_model_config(raw.get("model", {}))
    training_cfg = load_training_config(raw.get("training", {}))
    eval_cfg = load_eval_config(raw.get("eval", {}))

    experiment_fields = {
        "experiment_name": raw.get("experiment_name", path.stem),
        "run_id": raw.get("run_id"),
        "data": data_cfg,
        "model": model_cfg,
        "training": training_cfg,
        "eval": eval_cfg,
    }
    return ExperimentConfig.model_validate(experiment_fields)

