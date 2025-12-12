"""Sensor-agnostic ingest that turns spectra + metadata into model tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn

from alchemi.config.core import IngestConfig, ModelConfig
from alchemi.types import QuantityKind, Sample


def _as_tensor(
    x: Tensor | Sequence[float] | None, device: torch.device, dtype: torch.dtype
) -> Tensor:
    if x is None:
        return torch.tensor([], device=device, dtype=dtype)
    if isinstance(x, Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


@dataclass
class IngestOutput:
    """Container returned by :class:`AnySensorIngest`."""

    tokens: Tensor
    spectral_groups: int
    spatial_shape: tuple[int, int] | None


class AnySensorIngest(nn.Module):
    """Tokenises arbitrary sensor spectra into backbone-ready embeddings."""

    def __init__(
        self,
        d_model: int = 256,
        group_size: int = 1,
        patch_size: int = 1,
        quantity_kinds: Iterable[QuantityKind] = QuantityKind,
        max_sensors: int = 4,
        include_srf: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.group_size = group_size
        self.patch_size = patch_size
        self.include_srf = include_srf

        self.value_proj = nn.Linear(1, d_model)
        self.wavelength_proj = nn.Linear(1, d_model)
        self.bandwidth_proj = nn.Linear(1, d_model)
        self.srf_proj = nn.Sequential(
            nn.Linear(64, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.quantity_embed = nn.Embedding(len(tuple(quantity_kinds)), d_model)
        self.sensor_embed = nn.Embedding(max_sensors, d_model)
        self.out_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @classmethod
    def from_config(cls, cfg: ModelConfig | IngestConfig) -> "AnySensorIngest":
        if isinstance(cfg, ModelConfig):
            ingest_cfg = cfg.ingest
        else:
            ingest_cfg = cfg
        return cls(
            d_model=ingest_cfg.patch_embed_dim,
            group_size=max(1, ingest_cfg.spectral_bins // max(ingest_cfg.spectral_bins, 1)),
            patch_size=1,
            max_sensors=ingest_cfg.max_sensors,
            include_srf=True,
            dropout=ingest_cfg.dropout,
        )

    def forward(
        self,
        sample: Sample | Tensor,
        *,
        wavelengths: Tensor | Sequence[float] | None = None,
        bandwidths: Tensor | Sequence[float] | None = None,
        srf: Tensor | None = None,
        quantity: Tensor | Sequence[int] | QuantityKind | str | None = None,
        sensor_ids: Tensor | Sequence[int] | None = None,
    ) -> IngestOutput:
        """
        Parameters
        ----------
        sample:
            Either a :class:`Sample` or raw tensor shaped ``(B, Bands)`` or
            ``(B, H, W, Bands)``.
        wavelengths / bandwidths:
            Optional metadata arrays (nm) aligned with the spectral dimension.
        srf:
            Spectral response function per band (``B, Bands, S``). Compressed with
            a small MLP before fusion.
        quantity:
            Quantity kind indices or enum values.
        sensor_ids:
            Integer IDs identifying the sensor source per batch element.
        """

        values, wl, bw, qty, sensor = self._extract(
            sample, wavelengths, bandwidths, quantity, sensor_ids
        )
        batch = values.shape[0]

        if values.dim() == 2:
            spatial_shape: tuple[int, int] | None = None
            flat = values.unsqueeze(1)  # (B, 1, Bands)
        else:
            # (B, H, W, Bands) -> (B, H*W, Bands)
            spatial_shape = (values.shape[1], values.shape[2])
            flat = values.view(batch, -1, values.shape[-1])

        band_tokens = self._encode_band_tokens(flat, wl, bw, srf, qty, sensor)
        grouped = self._group_spectrally(band_tokens)
        tokens = self.out_norm(self.dropout(grouped))
        return IngestOutput(
            tokens=tokens, spectral_groups=grouped.shape[2], spatial_shape=spatial_shape
        )

    def _extract(
        self,
        sample: Sample | Tensor,
        wavelengths: Tensor | Sequence[float] | None,
        bandwidths: Tensor | Sequence[float] | None,
        quantity: Tensor | Sequence[int] | QuantityKind | str | None,
        sensor_ids: Tensor | Sequence[int] | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if isinstance(sample, Sample):
            if hasattr(sample, "spectrum"):
                values = torch.as_tensor(sample.spectrum.values)
                wl = torch.as_tensor(sample.spectrum.wavelength_nm)
                widths = None
                if getattr(sample, "band_meta", None) is not None:
                    widths = getattr(sample.band_meta, "width_nm", None)
                bw = torch.as_tensor(widths) if widths is not None else torch.zeros_like(wl)
                qty = torch.tensor(
                    [int(sample.spectrum.kind.value != QuantityKind.RADIANCE.value)]
                )
                sensor_id = getattr(sample, "sensor_id", 0)
                if isinstance(sensor_id, str):
                    sensor_id = 0
                sensor = torch.tensor([sensor_id], dtype=torch.long)
            else:
                values = torch.as_tensor(sample.values)
                wl = torch.as_tensor(sample.wavelengths)
                bw = torch.as_tensor(getattr(sample, "bandwidths", torch.zeros_like(wl)))
                qty = torch.tensor([int(sample.quantity_kind.value != QuantityKind.RADIANCE.value)])
                sensor = torch.tensor([getattr(sample, "sensor_id", 0)], dtype=torch.long)
        else:
            values = sample
            wl = _as_tensor(wavelengths, values.device, values.dtype)
            bw = _as_tensor(bandwidths, values.device, values.dtype)
            if isinstance(quantity, Tensor):
                qty = quantity.to(values.device)
            elif isinstance(quantity, QuantityKind):
                qty = torch.tensor([list(QuantityKind).index(quantity)], device=values.device)
            elif isinstance(quantity, str):
                qty = torch.tensor(
                    [list(QuantityKind).index(QuantityKind(quantity))], device=values.device
                )
            else:
                qty = torch.zeros(values.shape[0], device=values.device, dtype=torch.long)
            sensor = (
                sensor_ids.to(values.device)
                if isinstance(sensor_ids, Tensor)
                else torch.tensor(sensor_ids or 0, device=values.device)
            )

        wl = (
            wl
            if wl.numel() > 0
            else torch.linspace(0, 1, values.shape[-1], device=values.device, dtype=values.dtype)
        )
        bw = bw if bw.numel() > 0 else torch.full_like(wl, 1.0 / max(values.shape[-1], 1))
        values = values.to(dtype=torch.float32)
        wl = wl.to(dtype=torch.float32)
        bw = bw.to(dtype=torch.float32)
        if values.dim() == 1:
            values = values.unsqueeze(0)
        if wl.dim() == 1:
            wl = wl.unsqueeze(0)
        if bw.dim() == 1:
            bw = bw.unsqueeze(0)
        if wl.dim() == 1:
            wl = wl.unsqueeze(0).expand(values.shape[0], -1)
        if bw.dim() == 1:
            bw = bw.unsqueeze(0).expand(values.shape[0], -1)
        if sensor.dim() == 0:
            sensor = sensor.expand(values.shape[0])
        if qty.dim() == 0:
            qty = qty.expand(values.shape[0])
        return values, wl, bw, qty.long(), sensor.long()

    def _encode_band_tokens(
        self,
        flat: Tensor,
        wavelengths: Tensor,
        bandwidths: Tensor,
        srf: Tensor | None,
        quantity: Tensor,
        sensor_ids: Tensor,
    ) -> Tensor:
        B, T, Bands = flat.shape
        wl = wavelengths.view(B, 1, Bands, 1).expand(-1, T, -1, -1)
        bw = bandwidths.view(B, 1, Bands, 1).expand(-1, T, -1, -1)
        values = flat.unsqueeze(-1)

        value_emb = self.value_proj(values)
        wl_emb = self.wavelength_proj(wl)
        bw_emb = self.bandwidth_proj(bw)

        if srf is not None and self.include_srf:
            if srf.dim() == 2:
                srf = srf.unsqueeze(0).expand(B, -1, -1)
            srf_vec = srf.unsqueeze(1).expand(-1, T, -1, -1)
            srf_emb = self.srf_proj(srf_vec)
        else:
            srf_emb = torch.zeros_like(value_emb)

        quantity_emb = self.quantity_embed(quantity).view(B, 1, 1, -1)
        quantity_emb = quantity_emb.expand(-1, T, Bands, -1)
        sensor_emb = self.sensor_embed(sensor_ids).view(B, 1, 1, -1).expand(-1, T, Bands, -1)

        token = value_emb + wl_emb + bw_emb + srf_emb + quantity_emb + sensor_emb
        return token  # (B, T, Bands, D)

    def _group_spectrally(self, tokens: Tensor) -> Tensor:
        B, T, Bands, D = tokens.shape
        g = self.group_size
        if g <= 1:
            grouped = tokens.reshape(B, T * Bands, D)
        else:
            pad = (g - (Bands % g)) % g
            if pad:
                pad_tensor = torch.zeros(B, T, pad, D, device=tokens.device, dtype=tokens.dtype)
                tokens = torch.cat([tokens, pad_tensor], dim=2)
            grouped = tokens.view(B, T, -1, g, D).mean(dim=3)
            grouped = grouped.view(B, -1, D)
        return grouped
