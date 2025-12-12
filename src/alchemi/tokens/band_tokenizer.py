"""Tokenisation primitives for per-band spectral encodings.

``BandTokenizer`` consumes per-band spectra and augments each band with
positional and optional instrument metadata before projecting the features into
tokens. Inputs are expected to be one-dimensional spectra; batching is not
currently supported.

Key expectations
----------------
* ``values``: shape ``(C,)`` of per-band measurements.
* ``axis``: shape ``(C,)`` spectral coordinate matching ``values``.
* ``width``: optional scalar or array. If an array is provided it must be
  shape-compatible with ``axis`` (``(C,)``). Scalars are broadcast to ``C``.
* ``srf_row``: optional sensor-response features aligned to bands. Must have
  the same leading dimension as ``values``/``axis``; either ``(C,)`` or
  ``(C, S)`` where ``S`` is the number of SRF-derived features.

Units and normalisation
-----------------------
* ``axis_unit`` describes the provided coordinates and must be either
  ``"nm"`` (wavelength) or ``"cm-1"`` (wavenumber). Values are internally
  converted to the configured :class:`BandTokConfig.lambda_unit` before
  normalisation.
* ``value_norm`` controls how ``values`` are scaled: ``"none"`` leaves them
  unchanged, ``"per_spectrum_zscore"`` and ``"robust"`` compute statistics per
  input spectrum, and ``"global_zscore"`` expects precomputed statistics.
  When ``global_zscore`` is selected, :class:`ValueStats` must be supplied at
  construction time; otherwise a ``ValueError`` is raised when calling the
  tokenizer.
* ``include_width`` toggles whether band full-width-at-half-maximum metadata is
  incorporated (using defaults when ``width`` is omitted).
* ``include_srf_embed`` enables SRF feature projection when ``srf_row`` is
  supplied; SRF data is ignored when this flag is ``False``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from ..srf.utils import resolve_band_widths
from ..spectral.srf import SRFProvenance

AxisUnit = Literal["nm", "cm-1"]
ValueNorm = Literal["none", "per_spectrum_zscore", "global_zscore", "robust"]
FloatArray: TypeAlias = npt.NDArray[np.float64]


@dataclass(slots=True)
class ValueStats:
    """Global statistics used when applying ``global_zscore`` normalisation."""

    mean: FloatArray | Sequence[float] | float
    std: FloatArray | Sequence[float] | float


@dataclass(slots=True)
class BandTokConfig:
    """Configuration parameters describing the band tokeniser behaviour."""

    n_fourier_frequencies: int = 6
    value_norm: ValueNorm = "per_spectrum_zscore"
    lambda_unit: AxisUnit = "nm"
    include_width: bool = True
    include_srf_embed: bool = False
    include_srf_provenance: bool = False
    token_dim: int = 0
    axis_norm: Literal["zscore", "minmax"] = "zscore"
    srf_embed_dim: int = 8
    srf_provenance_embed_dim: int = 3
    projection_seed: int = 7
    sensor_id: str | None = None
    epsilon: float = 1e-6


@dataclass(slots=True)
class TokenMeta:
    """Metadata captured alongside band tokens."""

    axis_unit: AxisUnit
    value_norm: ValueNorm
    token_dim: int
    n_fourier_frequencies: int
    include_width: bool
    include_srf_embed: bool
    include_srf_provenance: bool
    invalid_mask: np.ndarray
    used_default_width: bool


@dataclass(slots=True)
class Tokens:
    """Container for band tokens and pooled summary."""

    bands: np.ndarray
    pooled: np.ndarray
    meta: TokenMeta


class BandTokenizer:
    """Transform spectral bands into fixed-dimensional token representations."""

    def __init__(
        self,
        config: BandTokConfig | None = None,
        stats: ValueStats | None = None,
    ) -> None:
        self._config = config or BandTokConfig()
        self._stats = stats
        self._feature_proj: FloatArray | None = None
        self._srf_proj: FloatArray | None = None
        self._srf_prov_embed: FloatArray | None = None

    @property
    def config(self) -> BandTokConfig:
        return self._config

    def __call__(
        self,
        values: npt.ArrayLike,
        axis: npt.ArrayLike,
        *,
        axis_unit: AxisUnit,
        width: npt.ArrayLike | float | None = None,
        width_from_default: npt.ArrayLike | bool | None = None,
        srf_row: np.ndarray | None = None,
        srf_provenance: npt.ArrayLike | str | None = None,
    ) -> Tokens:
        cfg = self._config

        vals = np.asarray(values, dtype=np.float64)
        coords = np.asarray(axis, dtype=np.float64)
        if vals.ndim != 1 or coords.ndim != 1:
            msg = "values and axis must be one-dimensional (shape (C,))"
            raise ValueError(msg)
        if vals.shape != coords.shape:
            msg = (
                "values and axis must have matching lengths; "
                f"got values shape {vals.shape} and axis shape {coords.shape}"
            )
            raise ValueError(msg)
        channels = vals.shape[0]

        if cfg.include_srf_embed and srf_row is not None:
            srf_arr = np.asarray(srf_row)
            if srf_arr.shape[0] != channels:
                msg = (
                    "srf_row must align with the spectral axis; "
                    f"expected {channels} rows, got {srf_arr.shape[0]}"
                )
                raise ValueError(msg)

        axis_nm = self._to_nm(coords, axis_unit)
        axis_target = self._convert_axis(coords, axis_unit, cfg.lambda_unit)

        norm_values, invalid_mask = self._normalise_values(vals)
        axis_features = self._normalise_axis(axis_target)
        fourier = self._fourier_features(axis_features)

        width_features = None
        used_default = False
        if cfg.include_width:
            width_nm, used_default = self._resolve_width(
                width, axis_nm, axis_unit, width_from_default=width_from_default
            )
            width_features = self._normalise_width(width_nm)

        srf_features = None
        if cfg.include_srf_embed and srf_row is not None:
            srf_features = self._compress_srf_features(srf_row)

        provenance_features = None
        if cfg.include_srf_provenance and srf_provenance is not None:
            provenance_features = self._encode_provenance(srf_provenance, channels)

        features = [norm_values[:, None], axis_features[:, None], fourier]
        if width_features is not None:
            features.append(width_features[:, None])
        if srf_features is not None:
            features.append(srf_features)
        if provenance_features is not None:
            features.append(provenance_features)
        raw = np.concatenate([feat for feat in features if feat.size], axis=1).astype(
            np.float64, copy=False
        )

        bands = self._project_features(raw).astype(np.float32, copy=False)
        pooled = np.nanmean(bands, axis=0).astype(np.float32, copy=False)

        meta = TokenMeta(
            axis_unit=axis_unit,
            value_norm=cfg.value_norm,
            token_dim=bands.shape[1],
            n_fourier_frequencies=cfg.n_fourier_frequencies,
            include_width=cfg.include_width,
            include_srf_embed=cfg.include_srf_embed,
            include_srf_provenance=cfg.include_srf_provenance,
            invalid_mask=invalid_mask,
            used_default_width=used_default,
        )
        return Tokens(bands=bands, pooled=pooled, meta=meta)

    def _normalise_values(self, values: FloatArray) -> tuple[FloatArray, np.ndarray]:
        cfg = self._config
        eps = cfg.epsilon
        mask = np.isfinite(values)
        if not np.any(mask):
            return np.zeros_like(values), ~mask

        if cfg.value_norm == "none":
            cleaned = np.where(mask, values, 0.0).astype(np.float64, copy=False)
            return cleaned, ~mask

        cleaned = np.where(mask, values, np.nan).astype(np.float64, copy=False)

        if cfg.value_norm == "per_spectrum_zscore":
            mean_scalar = float(np.nanmean(cleaned))
            std_scalar = float(np.nanstd(cleaned))
            mean = np.full(values.shape, mean_scalar, dtype=np.float64)
            std = np.full(values.shape, std_scalar, dtype=np.float64)
        elif cfg.value_norm == "robust":
            median = float(np.nanmedian(cleaned))
            quantiles: FloatArray = np.nanpercentile(cleaned, [25.0, 75.0]).astype(
                np.float64, copy=False
            )
            q25 = float(quantiles[0])
            q75 = float(quantiles[1])
            mean = np.full(values.shape, median, dtype=np.float64)
            std = np.full(values.shape, float(q75 - q25), dtype=np.float64)
        elif cfg.value_norm == "global_zscore":
            if self._stats is None:
                raise ValueError("global_zscore requested but ValueStats were not provided")
            mean = self._broadcast(values.shape, self._stats.mean)
            std = self._broadcast(values.shape, self._stats.std)
        else:
            raise ValueError(f"Unsupported value normalisation '{cfg.value_norm}'")

        std = np.maximum(std, eps)
        normed = (cleaned - mean) / std
        normed[~mask] = 0.0
        return normed.astype(np.float64, copy=False), ~mask

    def _normalise_axis(self, axis: FloatArray) -> FloatArray:
        cfg = self._config
        eps = cfg.epsilon
        if cfg.axis_norm == "zscore":
            mean = float(np.mean(axis))
            std = float(np.std(axis))
            std = max(std, eps)
            return ((axis - mean) / std).astype(np.float64, copy=False)
        if cfg.axis_norm == "minmax":
            lo = float(np.min(axis))
            hi = float(np.max(axis))
            span = max(hi - lo, eps)
            return ((axis - lo) / span).astype(np.float64, copy=False)
        raise ValueError(f"Unsupported axis_norm '{cfg.axis_norm}'")

    def _fourier_features(self, axis_norm: FloatArray) -> FloatArray:
        n_freq = max(int(self._config.n_fourier_frequencies), 0)
        if n_freq == 0:
            return np.zeros((axis_norm.shape[0], 0), dtype=np.float64)
        freq_indices: FloatArray = np.arange(n_freq, dtype=np.float64)
        scales = np.pi * (2.0**freq_indices)
        angles = np.outer(axis_norm, scales)
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=1).astype(
            np.float64, copy=False
        )

    def _resolve_width(
        self,
        width: Sequence[float] | float | None,
        axis_nm: FloatArray,
        axis_unit: AxisUnit,
        *,
        width_from_default: Sequence[bool] | bool | None = None,
    ) -> tuple[FloatArray, bool]:
        cfg = self._config
        if width is not None:
            arr = np.asarray(width, dtype=np.float64)
            if arr.ndim == 0:
                arr = np.full(axis_nm.shape, float(arr), dtype=np.float64)
            if arr.shape != axis_nm.shape:
                msg = (
                    "width must align with the spectral axis; "
                    f"expected shape {axis_nm.shape}, got {arr.shape}"
                )
                raise ValueError(msg)
            if axis_unit == "cm-1":
                # Convert from wavenumber (cm^-1) to nanometres via derivative dλ/dnu.
                nu = 1.0e7 / axis_nm
                arr = (1.0e7 / (nu**2)) * arr
            if width_from_default is None:
                default_mask = np.zeros_like(arr, dtype=bool)
            else:
                default_mask = np.asarray(width_from_default, dtype=bool)
                if default_mask.ndim == 0:
                    default_mask = np.full(arr.shape, bool(default_mask), dtype=bool)
                if default_mask.shape != arr.shape:
                    raise ValueError(
                        "width_from_default must be broadcastable to the spectral axis"
                    )
            return arr.astype(np.float64, copy=False), bool(np.any(default_mask))

        resolved, default_mask, _ = resolve_band_widths(
            cfg.sensor_id, axis_nm, registry=None, srf=None
        )
        return resolved.astype(np.float64, copy=False), bool(np.any(default_mask))

    def _normalise_width(self, width_nm: FloatArray) -> FloatArray:
        eps = self._config.epsilon
        width = np.asarray(width_nm, dtype=np.float64)
        width[width <= 0] = np.nan
        fallback = float(np.nanmean(width)) if np.any(np.isfinite(width)) else 1.0
        logw = np.log1p(np.nan_to_num(width, nan=fallback))
        logw -= np.nanmean(logw)
        std_val = float(np.nanstd(logw))
        std = max(std_val, eps)
        return (logw / std).astype(np.float64, copy=False)

    def _compress_srf_features(self, srf_features: np.ndarray) -> FloatArray:
        cfg = self._config
        rows = np.asarray(srf_features, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows[:, None]
        if rows.shape[1] == cfg.srf_embed_dim:
            return rows.astype(np.float64, copy=False)
        proj = self._srf_proj
        if proj is None or proj.shape[0] != rows.shape[1]:
            rng = np.random.default_rng(cfg.projection_seed + 1)
            proj = rng.standard_normal((rows.shape[1], cfg.srf_embed_dim))
            norms = np.linalg.norm(proj, axis=0, keepdims=True) + cfg.epsilon
            proj = proj / norms
            self._srf_proj = proj
        return (rows @ proj).astype(np.float64, copy=False)

    def _encode_provenance(
        self, provenance: npt.ArrayLike | str, channels: int
    ) -> FloatArray:
        cfg = self._config
        prov_arr = np.asarray(provenance, dtype=object)
        if prov_arr.ndim == 0:
            prov_arr = np.full(channels, prov_arr.item(), dtype=object)
        prov_arr = prov_arr.reshape(-1)
        if prov_arr.shape[0] != channels:
            msg = f"srf_provenance must align with spectral axis; expected {channels}, got {prov_arr.shape[0]}"
            raise ValueError(msg)

        embed = self._srf_prov_embed
        if embed is None or embed.shape[1] != cfg.srf_provenance_embed_dim:
            rng = np.random.default_rng(cfg.projection_seed + 2)
            embed = rng.standard_normal((len(SRFProvenance), cfg.srf_provenance_embed_dim))
            self._srf_prov_embed = embed

        idx_map = {prov.value: idx for idx, prov in enumerate(SRFProvenance)}
        encoded = np.empty((channels, cfg.srf_provenance_embed_dim), dtype=np.float64)
        for i, label in enumerate(prov_arr):
            key = str(label).lower()
            idx = idx_map.get(key, idx_map[SRFProvenance.NONE.value])
            encoded[i] = embed[idx]
        return encoded.astype(np.float64, copy=False)

    def _project_features(self, features: FloatArray) -> FloatArray:
        cfg = self._config
        if cfg.token_dim <= 0 or features.shape[1] == cfg.token_dim:
            return features
        proj = self._feature_proj
        if proj is None or proj.shape[0] != features.shape[1]:
            rng = np.random.default_rng(cfg.projection_seed)
            proj = rng.standard_normal((features.shape[1], cfg.token_dim))
            norms = np.linalg.norm(proj, axis=0, keepdims=True) + cfg.epsilon
            proj = proj / norms
            self._feature_proj = proj
        return (features @ proj).astype(np.float64, copy=False)

    @staticmethod
    def _to_nm(axis: np.ndarray, axis_unit: AxisUnit) -> FloatArray:
        if axis_unit == "nm":
            return axis.astype(np.float64, copy=False)
        if axis_unit == "cm-1":
            if np.any(axis <= 0):
                raise ValueError("Wavenumber axis must contain positive values")
            return (1.0e7 / axis).astype(np.float64, copy=False)
        raise ValueError(f"Unsupported axis unit '{axis_unit}'")

    def _convert_axis(
        self,
        axis: np.ndarray,
        src_unit: AxisUnit,
        dst_unit: AxisUnit,
    ) -> FloatArray:
        if src_unit == dst_unit:
            return axis.astype(np.float64, copy=False)
        if src_unit == "nm" and dst_unit == "cm-1":
            if np.any(axis <= 0):
                raise ValueError("Wavelength axis must contain positive values")
            return (1.0e7 / axis).astype(np.float64, copy=False)
        if src_unit == "cm-1" and dst_unit == "nm":
            return self._to_nm(axis, "cm-1")
        raise ValueError(f"Unsupported axis conversion {src_unit!r} -> {dst_unit!r}")

    @staticmethod
    def _broadcast(
        shape: tuple[int, ...], values: FloatArray | Sequence[float] | float
    ) -> FloatArray:
        arr = np.asarray(values, dtype=np.float64)
        try:
            broadcasted = np.broadcast_to(arr, shape)
            return np.asarray(broadcasted, dtype=np.float64)
        except ValueError as exc:  # pragma: no cover - defensive
            msg = "Provided statistics could not be broadcast to spectrum shape"
            raise ValueError(msg) from exc


__all__ = ["AxisUnit", "BandTokConfig", "BandTokenizer", "TokenMeta", "Tokens", "ValueStats"]
