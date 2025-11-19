"""Cycle-consistency reconstruction heads for alignment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    cycle_raw: bool = False
    cycle_continuum: bool = False
    slope_reg: bool = False
    continuum_weight: float = 1.0
    slope_weight: float = 1.0
    continuum_window_nm: tuple[float, float] | None = None


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
        self,
        z_lab: torch.Tensor,
        sensor_tokens: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
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
                msg = (
                    "sensor_tokens are required on the first call to initialize the "
                    "reconstruction head"
                )
                raise ValueError(msg)
            sensor_targets, _ = self._resolve_sensor_targets(sensor_tokens)
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
        self,
        z_sensor: torch.Tensor,
        lab_tokens: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
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
                msg = (
                    "lab_tokens are required on the first call to initialize the "
                    "reconstruction head"
                )
                raise ValueError(msg)
            # New behavior: support mappings and axes via _resolve_lab_targets
            lab_targets, _ = self._resolve_lab_targets(lab_tokens)
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
        lab_tokens: torch.Tensor | Mapping[str, torch.Tensor],
        sensor_tokens: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the bidirectional cycle reconstruction loss."""

        if not self.enabled:
            zero = torch.zeros((), device=z_lab.device, dtype=z_lab.dtype)
            return zero, {}

        # New behavior: axis-aware resolution from mappings OR tensors
        sensor_targets, sensor_axis = self._resolve_sensor_targets(sensor_tokens)
        lab_targets, lab_axis = self._resolve_lab_targets(lab_tokens)

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

        total = torch.zeros((), device=z_lab.device, dtype=z_lab.dtype)
        breakdown: dict[str, torch.Tensor] = {}

        if self.config.cycle_raw:
            sensor_l2 = self._mse(pred_sensor, sensor_targets)
            lab_l2 = self._mse(pred_lab, lab_targets)
            sensor_sam = self.sam_loss(pred_sensor, sensor_targets)
            lab_sam = self.sam_loss(pred_lab, lab_targets)
            raw_loss = self.config.l2_weight * (sensor_l2 + lab_l2) + self.config.sam_weight * (
                sensor_sam + lab_sam
            )
            total = total + raw_loss
            breakdown.update(
                {
                    "sensor_l2": sensor_l2.detach(),
                    "lab_l2": lab_l2.detach(),
                    "sensor_sam": sensor_sam.detach(),
                    "lab_sam": lab_sam.detach(),
                }
            )

        if self.config.cycle_continuum:
            cont_losses = self._continuum_losses(
                pred_lab,
                lab_targets,
                lab_axis,
                pred_sensor,
                sensor_targets,
                sensor_axis,
            )
            if cont_losses is not None:
                cont_loss, cont_breakdown = cont_losses
                total = total + self.config.continuum_weight * cont_loss
                breakdown.update(cont_breakdown)

        if self.config.slope_reg:
            slope_losses = self._slope_losses(
                pred_lab,
                lab_targets,
                lab_axis,
                pred_sensor,
                sensor_targets,
                sensor_axis,
            )
            if slope_losses is not None:
                slope_loss, slope_breakdown = slope_losses
                total = total + self.config.slope_weight * slope_loss
                breakdown.update(slope_breakdown)

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
                msg = (
                    "Context features are required once a positive context dimension is configured."
                )
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(sensor_tokens, Mapping):
            candidate_keys = ["bt", "radiance", "values"]
            preferred = "bt" if self.config.use_brightness_temperature else "radiance"
            for key in (preferred, *candidate_keys):
                if key in sensor_tokens:
                    axis_payload = sensor_tokens.get("wavelengths_nm")
                    if axis_payload is None:
                        axis_payload = sensor_tokens.get("axis")
                    if axis_payload is None:
                        axis_payload = sensor_tokens.get("axis_nm")
                    axis = self._ensure_axis_tensor(
                        axis_payload,
                        sensor_tokens[key],
                    )
                    return sensor_tokens[key], axis
            available = ", ".join(sensor_tokens.keys())
            msg = f"No suitable sensor target found in mapping. Available keys: {available}"
            raise KeyError(msg)
        if not isinstance(sensor_tokens, torch.Tensor):
            msg = "sensor_tokens must be a tensor or mapping"
            raise TypeError(msg)
        return sensor_tokens, None

    def _resolve_lab_targets(
        self, lab_tokens: torch.Tensor | Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(lab_tokens, Mapping):
            for key in ("values", "reflectance", "radiance"):
                if key in lab_tokens:
                    axis_payload = lab_tokens.get("wavelengths_nm")
                    if axis_payload is None:
                        axis_payload = lab_tokens.get("axis")
                    if axis_payload is None:
                        axis_payload = lab_tokens.get("axis_nm")
                    axis = self._ensure_axis_tensor(
                        axis_payload,
                        lab_tokens[key],
                    )
                    return lab_tokens[key], axis
            available = ", ".join(lab_tokens.keys())
            msg = f"No suitable lab target found in mapping. Available keys: {available}"
            raise KeyError(msg)
        if not isinstance(lab_tokens, torch.Tensor):
            msg = "lab_tokens must be a tensor or mapping"
            raise TypeError(msg)
        return lab_tokens, None

    def _ensure_axis_tensor(
        self, axis: torch.Tensor | Mapping[str, torch.Tensor] | None, reference: torch.Tensor
    ) -> torch.Tensor | None:
        if axis is None:
            return None
        if isinstance(axis, Mapping):  # pragma: no cover - defensive guard
            msg = "Axis payloads must be tensors or arrays"
            raise TypeError(msg)
        tensor = torch.as_tensor(axis, device=reference.device, dtype=reference.dtype)
        if tensor.dim() != 1:
            raise ValueError("Wavelength axes must be 1-D")
        return tensor

    def _continuum_losses(
        self,
        pred_lab: torch.Tensor,
        lab_targets: torch.Tensor,
        lab_axis: torch.Tensor | None,
        pred_sensor: torch.Tensor,
        sensor_targets: torch.Tensor,
        sensor_axis: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | None:
        if lab_axis is None or sensor_axis is None:
            return None
        lab_removed, _ = _continuum_removed(lab_targets, lab_axis, self.config)
        pred_lab_removed, _ = _continuum_removed(pred_lab, lab_axis, self.config)
        sensor_removed, _ = _continuum_removed(sensor_targets, sensor_axis, self.config)
        pred_sensor_removed, _ = _continuum_removed(pred_sensor, sensor_axis, self.config)

        lab_mse = self._mse(pred_lab_removed, lab_removed)
        sensor_mse = self._mse(pred_sensor_removed, sensor_removed)
        lab_sam = self.sam_loss(pred_lab_removed, lab_removed)
        sensor_sam = self.sam_loss(pred_sensor_removed, sensor_removed)
        total = lab_mse + sensor_mse + lab_sam + sensor_sam
        breakdown = {
            "lab_cont_mse": lab_mse.detach(),
            "sensor_cont_mse": sensor_mse.detach(),
            "lab_cont_sam": lab_sam.detach(),
            "sensor_cont_sam": sensor_sam.detach(),
        }
        return total, breakdown

    def _slope_losses(
        self,
        pred_lab: torch.Tensor,
        lab_targets: torch.Tensor,
        lab_axis: torch.Tensor | None,
        pred_sensor: torch.Tensor,
        sensor_targets: torch.Tensor,
        sensor_axis: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | None:
        if lab_axis is None or sensor_axis is None:
            return None
        lab_slope = _spectral_slope(lab_targets, lab_axis)
        pred_lab_slope = _spectral_slope(pred_lab, lab_axis)
        sensor_slope = _spectral_slope(sensor_targets, sensor_axis)
        pred_sensor_slope = _spectral_slope(pred_sensor, sensor_axis)

        lab_loss = F.mse_loss(pred_lab_slope, lab_slope)
        sensor_loss = F.mse_loss(pred_sensor_slope, sensor_slope)
        total = lab_loss + sensor_loss
        breakdown = {
            "lab_slope": lab_loss.detach(),
            "sensor_slope": sensor_loss.detach(),
        }
        return total, breakdown


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
        lab_tokens: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
        sensor_tokens: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        losses["infonce"] = self.info_nce(z_lab, z_sensor)

        if self.cycle.enabled and lab_tokens is not None and sensor_tokens is not None:
            cycle_loss, breakdown = self.cycle.cycle_loss(
                z_lab,
                z_sensor,
                lab_tokens,
                sensor_tokens,
            )
            losses["cycle"] = cycle_loss
            for key, value in breakdown.items():
                losses[f"cycle_{key}"] = value
        return losses

    def reconstruct_sensor_from_lab(
        self,
        z_lab: torch.Tensor,
        sensor_tokens: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return self.cycle.reconstruct_sensor_from_lab(z_lab, sensor_tokens)

    def reconstruct_lab_from_sensor(
        self,
        z_sensor: torch.Tensor,
        lab_tokens: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return self.cycle.reconstruct_lab_from_sensor(z_sensor, lab_tokens)


def _continuum_removed(
    values: torch.Tensor, axis: torch.Tensor, config: CycleConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    axis = axis.to(device=values.device, dtype=values.dtype)
    if axis.dim() != 1:
        raise ValueError("Wavelength axis must be 1-D for continuum removal")
    if axis.numel() < 2:
        raise ValueError("Wavelength axis must contain at least two samples")
    default_window = (float(axis[0].item()), float(axis[-1].item()))
    left_nm, right_nm = config.continuum_window_nm or default_window
    left_nm_t = torch.as_tensor(left_nm, device=axis.device, dtype=axis.dtype)
    right_nm_t = torch.as_tensor(right_nm, device=axis.device, dtype=axis.dtype)
    left_vals = _interp_axis(values, axis, left_nm_t)
    right_vals = _interp_axis(values, axis, right_nm_t)
    denom = torch.clamp(right_nm_t - left_nm_t, min=1e-6)
    slope = (right_vals - left_vals) / denom
    axis_row = axis.view(1, -1)
    continuum = left_vals + slope * (axis_row - left_nm_t)
    continuum = continuum.clamp_min(1e-6)
    removed = values / continuum
    removed = torch.nan_to_num(removed)
    return removed, continuum


def _interp_axis(values: torch.Tensor, axis: torch.Tensor, target_nm: torch.Tensor) -> torch.Tensor:
    idx = torch.searchsorted(axis, target_nm)
    idx = torch.clamp(idx, 1, axis.numel() - 1)
    lower = idx - 1
    upper = idx
    wl_lower = axis[lower]
    wl_upper = axis[upper]
    frac = (target_nm - wl_lower) / torch.clamp(wl_upper - wl_lower, min=1e-6)
    lower_vals = values[..., lower]
    upper_vals = values[..., upper]
    interp = lower_vals + frac * (upper_vals - lower_vals)
    return interp.unsqueeze(-1)


def _spectral_slope(values: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    axis = axis.to(device=values.device, dtype=values.dtype)
    if axis.numel() < 2:
        raise ValueError("Wavelength axis must contain at least two samples")
    diffs = values[..., 1:] - values[..., :-1]
    denom = torch.clamp(axis[1:] - axis[:-1], min=1e-6)
    denom = denom.view(1, -1)
    slopes = diffs / denom
    return torch.nan_to_num(slopes)
