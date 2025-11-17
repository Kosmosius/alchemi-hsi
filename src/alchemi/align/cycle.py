"""Cycle-consistency reconstruction heads for alignment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn

from alchemi.losses import InfoNCELoss, SAMLoss


@dataclass
class CycleConfig:
    """Configuration for optional cycle-consistency reconstruction heads."""

    enabled: bool = False
    hidden_dim: int | None = None
    num_layers: int = 2
    l2_weight: float = 1.0
    sam_weight: float = 1.0
    sensor_context_dim: int = 0
    lab_context_dim: int = 0
    use_brightness_temperature: bool = False


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int | None = None,
    *,
    num_layers: int = 2,
    activation: type[nn.Module] = nn.GELU,
) -> nn.Sequential:
    if num_layers < 1:
        msg = "num_layers must be >= 1"
        raise ValueError(msg)
    hidden_dim = hidden_dim or max(input_dim, output_dim)
    dims = [input_dim]
    if num_layers > 1:
        dims.extend([hidden_dim] * (num_layers - 1))
    dims.append(output_dim)
    layers: list[nn.Module] = []
    for idx in range(len(dims) - 1):
        layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class CycleReconstructionHeads(nn.Module):
    """Paired reconstruction heads enforcing cycle consistency between modalities."""

    def __init__(self, lab_dim: int, sensor_dim: int, config: CycleConfig | None = None):
        super().__init__()
        self.config = config or CycleConfig()
        self.enabled = self.config.enabled
        self.lab_dim = lab_dim
        self.sensor_dim = sensor_dim
        self.sensor_context_dim = max(self.config.sensor_context_dim, 0)
        self.lab_context_dim = max(self.config.lab_context_dim, 0)
        self._sensor_context_cached: int | None = None
        self._lab_context_cached: int | None = None

        self.sam_loss = SAMLoss()
        self._mse = nn.MSELoss()

        self.lab_to_sensor: nn.Module | None = None
        self.sensor_to_lab: nn.Module | None = None
        self._lab_to_sensor_in_features: int | None = None
        self._sensor_to_lab_in_features: int | None = None

    def reconstruct_sensor_from_lab(
        self, z_lab: torch.Tensor, sensor_tokens: torch.Tensor | Mapping[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Predict sensor-space values from lab embeddings."""

        self._ensure_enabled()
        inputs = self._concat_with_context(
            z_lab,
            sensor_tokens,
            self.sensor_context_dim,
            cache_attr="_sensor_context_cached",
        )
        if self.lab_to_sensor is None:
            if sensor_tokens is None:
                msg = "sensor_tokens are required on the first call to initialize the reconstruction head"
                raise ValueError(msg)
            sensor_targets = self._resolve_sensor_targets(sensor_tokens)
            self.lab_to_sensor = _build_mlp(
                inputs.shape[-1],
                sensor_targets.shape[-1],
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
            ).to(device=inputs.device, dtype=inputs.dtype)
            self._lab_to_sensor_in_features = inputs.shape[-1]
        elif inputs.shape[-1] != self._lab_to_sensor_in_features:
            msg = (
                "Input dimensionality for lab->sensor reconstruction changed: "
                f"expected {self._lab_to_sensor_in_features}, got {inputs.shape[-1]}"
            )
            raise ValueError(msg)
        return self.lab_to_sensor(inputs)

    def reconstruct_lab_from_sensor(
        self, z_sensor: torch.Tensor, lab_tokens: torch.Tensor | Mapping[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Predict lab-space spectra from sensor embeddings."""

        self._ensure_enabled()
        inputs = self._concat_with_context(
            z_sensor,
            lab_tokens,
            self.lab_context_dim,
            cache_attr="_lab_context_cached",
        )
        if self.sensor_to_lab is None:
            if lab_tokens is None:
                msg = "lab_tokens are required on the first call to initialize the reconstruction head"
                raise ValueError(msg)
            lab_targets = lab_tokens
            self.sensor_to_lab = _build_mlp(
                inputs.shape[-1],
                lab_targets.shape[-1],
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
            ).to(device=inputs.device, dtype=inputs.dtype)
            self._sensor_to_lab_in_features = inputs.shape[-1]
        elif inputs.shape[-1] != self._sensor_to_lab_in_features:
            msg = (
                "Input dimensionality for sensor->lab reconstruction changed: "
                f"expected {self._sensor_to_lab_in_features}, got {inputs.shape[-1]}"
            )
            raise ValueError(msg)
        return self.sensor_to_lab(inputs)

    def cycle_loss(
        self,
        z_lab: torch.Tensor,
        z_sensor: torch.Tensor,
        lab_tokens: torch.Tensor,
        sensor_tokens: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the bidirectional cycle reconstruction loss."""

        if not self.enabled:
            zero = torch.zeros((), device=z_lab.device, dtype=z_lab.dtype)
            return zero, {}

        sensor_targets = self._resolve_sensor_targets(sensor_tokens)
        lab_targets = lab_tokens

        if sensor_targets.dim() != 2 or sensor_targets.shape[-1] != self.sensor_dim:
            msg = (
                "sensor target dimensionality does not match reconstruction head: "
                f"expected (?, {self.sensor_dim}), got {tuple(sensor_targets.shape)}"
            )
            raise ValueError(msg)
        if lab_targets.dim() != 2 or lab_targets.shape[-1] != self.lab_dim:
            msg = (
                "lab target dimensionality does not match reconstruction head: "
                f"expected (?, {self.lab_dim}), got {tuple(lab_targets.shape)}"
            )
            raise ValueError(msg)

        pred_sensor = self.reconstruct_sensor_from_lab(z_lab, sensor_tokens)
        pred_lab = self.reconstruct_lab_from_sensor(z_sensor, lab_tokens)

        sensor_l2 = self._mse(pred_sensor, sensor_targets)
        lab_l2 = self._mse(pred_lab, lab_targets)
        sensor_sam = self.sam_loss(pred_sensor, sensor_targets)
        lab_sam = self.sam_loss(pred_lab, lab_targets)

        total = self.config.l2_weight * (sensor_l2 + lab_l2) + self.config.sam_weight * (sensor_sam + lab_sam)
        breakdown = {
            "sensor_l2": sensor_l2.detach(),
            "lab_l2": lab_l2.detach(),
            "sensor_sam": sensor_sam.detach(),
            "lab_sam": lab_sam.detach(),
        }
        return total, breakdown

    def _ensure_enabled(self) -> None:
        if not self.enabled:
            msg = "Cycle reconstruction heads are disabled."
            raise RuntimeError(msg)

    def _concat_with_context(
        self,
        base: torch.Tensor,
        tokens: torch.Tensor | Mapping[str, torch.Tensor] | None,
        configured_dim: int,
        *,
        cache_attr: str,
    ) -> torch.Tensor:
        if base.dim() != 2:
            msg = "Inputs must be 2D tensors of shape (batch, features)."
            raise ValueError(msg)

        context: torch.Tensor | None
        context = None
        if isinstance(tokens, Mapping):
            context = tokens.get("context")
        elif isinstance(tokens, torch.Tensor):
            context = tokens
        elif tokens is not None:
            msg = "tokens must be a tensor, mapping, or None"
            raise TypeError(msg)

        if configured_dim == 0:
            return base

        if context is None:
            cached_dim = getattr(self, cache_attr)
            if cached_dim not in (0, None):
                msg = "Context features are required once a positive context dimension is configured."
                raise ValueError(msg)
            return base

        if context.dim() != 2:
            msg = "Context tensors must be 2D (batch, features)."
            raise ValueError(msg)

        cached_dim = getattr(self, cache_attr)
        if cached_dim is None:
            setattr(self, cache_attr, context.shape[-1])
            cached_dim = context.shape[-1]
        if configured_dim and context.shape[-1] < configured_dim:
            msg = (
                "Context tensor has fewer features than configured context dimension: "
                f"expected >= {configured_dim}, got {context.shape[-1]}"
            )
            raise ValueError(msg)

        use_dim = configured_dim or cached_dim or context.shape[-1]
        return torch.cat([base, context[..., :use_dim]], dim=-1)

    def _resolve_sensor_targets(
        self, sensor_tokens: torch.Tensor | Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(sensor_tokens, Mapping):
            candidate_keys = ["bt", "radiance", "values"]
            preferred = "bt" if self.config.use_brightness_temperature else "radiance"
            for key in (preferred, *candidate_keys):
                if key in sensor_tokens:
                    return sensor_tokens[key]
            available = ", ".join(sensor_tokens.keys())
            msg = f"No suitable sensor target found in mapping. Available keys: {available}"
            raise KeyError(msg)
        if not isinstance(sensor_tokens, torch.Tensor):
            msg = "sensor_tokens must be a tensor or mapping"
            raise TypeError(msg)
        return sensor_tokens


class CycleAlignment(nn.Module):
    """Alignment head combining InfoNCE with optional cycle reconstruction losses."""

    def __init__(
        self,
        lab_dim: int,
        sensor_dim: int,
        cycle_config: CycleConfig | None = None,
    ) -> None:
        super().__init__()
        self.info_nce = InfoNCELoss()
        self.cycle = CycleReconstructionHeads(lab_dim, sensor_dim, cycle_config)

    def forward(
        self,
        z_lab: torch.Tensor,
        z_sensor: torch.Tensor,
        lab_tokens: torch.Tensor | None = None,
        sensor_tokens: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        losses["infonce"] = self.info_nce(z_lab, z_sensor)

        if self.cycle.enabled and lab_tokens is not None and sensor_tokens is not None:
            cycle_loss, breakdown = self.cycle.cycle_loss(z_lab, z_sensor, lab_tokens, sensor_tokens)
            losses["cycle"] = cycle_loss
            for key, value in breakdown.items():
                losses[f"cycle_{key}"] = value
        return losses

    def reconstruct_sensor_from_lab(
        self, z_lab: torch.Tensor, sensor_tokens: torch.Tensor | Mapping[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        return self.cycle.reconstruct_sensor_from_lab(z_lab, sensor_tokens)

    def reconstruct_lab_from_sensor(
        self, z_sensor: torch.Tensor, lab_tokens: torch.Tensor | Mapping[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        return self.cycle.reconstruct_lab_from_sensor(z_sensor, lab_tokens)
