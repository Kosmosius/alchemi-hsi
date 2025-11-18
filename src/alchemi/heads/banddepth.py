"""Band depth prediction head."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

import torch
import yaml
from torch import Tensor, nn

TFunc = TypeVar("TFunc", bound=Callable[..., Tensor])


def _no_grad(func: TFunc) -> TFunc:
    """Typed alias of torch.no_grad for use as a decorator."""

    return cast(TFunc, torch.no_grad()(func))


@dataclass(frozen=True, slots=True)
class BandDefinition:
    """Definition of an absorption feature for band-depth estimation."""

    center_nm: float
    left_nm: float
    right_nm: float
    name: str | None = None

    def __post_init__(self) -> None:
        if self.right_nm <= self.left_nm:
            msg = "right_nm must be greater than left_nm"
            raise ValueError(msg)
        if not (self.left_nm <= self.center_nm <= self.right_nm):
            msg = "center_nm must lie between left_nm and right_nm"
            raise ValueError(msg)


def load_banddepth_config(path: str | Path) -> list[BandDefinition]:
    """Load band definitions from a YAML configuration file."""

    data = yaml.safe_load(Path(path).read_text())
    bands_raw: Iterable[dict[str, Any]] = []
    if isinstance(data, dict):
        raw = data.get("bands", [])
        if isinstance(raw, Iterable):
            bands_raw = cast(Iterable[dict[str, Any]], raw)
    definitions = [
        BandDefinition(
            center_nm=float(spec["center"]),
            left_nm=float(spec.get("left", spec["center"] - spec.get("width", 10.0))),
            right_nm=float(spec.get("right", spec["center"] + spec.get("width", 10.0))),
            name=spec.get("name"),
        )
        for spec in bands_raw
    ]
    if not definitions:
        msg = "banddepth config must define at least one band"
        raise ValueError(msg)
    return definitions


class BandDepthHead(nn.Module):
    """Predict continuum-removed band depths from pooled embeddings."""

    def __init__(
        self,
        embed_dim: int,
        bands: Sequence[BandDefinition],
        hidden_dim: int | None = None,
        loss: str = "l1",
    ) -> None:
        super().__init__()
        if not bands:
            msg = "BandDepthHead requires at least one band definition"
            raise ValueError(msg)
        self.bands = list(bands)
        out_dim = len(self.bands)
        if hidden_dim is not None and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.net = nn.Linear(embed_dim, out_dim)
        loss = loss.lower()
        if loss == "l1":
            self._loss = nn.L1Loss()
        elif loss in {"l2", "mse"}:
            self._loss = nn.MSELoss()
        else:
            msg = "loss must be one of {'l1', 'l2', 'mse'}"
            raise ValueError(msg)

    def forward(self, pooled: Tensor) -> Tensor:
        """Predict band depths from pooled embeddings."""

        return self.net(pooled)

    def loss(self, preds: Tensor, targets: Tensor) -> Tensor:
        """Compute the configured regression loss."""

        return self._loss(preds, targets)

    @_no_grad
    def compute_targets(
        self,
        wavelengths_nm: Tensor,
        reflectance: Tensor,
    ) -> Tensor:
        """Compute continuum-removed band depths for a batch of spectra."""

        if reflectance.dim() == 1:
            reflectance = reflectance.unsqueeze(0)
        wavelengths_nm = wavelengths_nm.to(reflectance.device, reflectance.dtype)
        n_spec = reflectance.shape[-1]
        if wavelengths_nm.numel() != n_spec:
            msg = "wavelength grid length must match reflectance dimension"
            raise ValueError(msg)

        depths: list[Tensor] = []
        for spec in self.bands:
            left_idx = _clamped_index(wavelengths_nm, spec.left_nm)
            right_idx = _clamped_index(wavelengths_nm, spec.right_nm)
            center_idx = _clamped_index(wavelengths_nm, spec.center_nm)

            left_ref = reflectance[..., left_idx]
            right_ref = reflectance[..., right_idx]
            slope = (right_ref - left_ref) / (spec.right_nm - spec.left_nm + 1e-12)
            cont_center = torch.clamp(left_ref + slope * (spec.center_nm - spec.left_nm), min=1e-6)
            center_ref = reflectance[..., center_idx]
            removed_center = torch.clamp(center_ref / cont_center, max=5.0)
            depths.append(1.0 - removed_center)
        return torch.stack(depths, dim=-1)


def _clamped_index(wavelengths_nm: Tensor, value: float) -> int:
    idx = torch.searchsorted(
        wavelengths_nm,
        torch.tensor(value, device=wavelengths_nm.device, dtype=wavelengths_nm.dtype),
    )
    i = int(idx.clamp(0, wavelengths_nm.numel() - 1).item())
    return i
