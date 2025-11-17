"""Tokenisation primitives for per-band spectral encodings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np

from ..srf.registry import SRFRegistry, get_srf

AxisUnit = Literal["nm", "cm-1"]


@dataclass(slots=True)
class BandTokConfig:
    """Configuration for :class:`BandTokenizer`.

    Parameters
    ----------
    value_norm:
        Normalisation strategy applied to band values. ``"zscore"`` expects
        pre-computed global statistics (``value_mean`` and ``value_std``), while
        ``"robust"`` performs per-spectrum median/IQR scaling.
    lambda_norm:
        Normalisation applied to wavelength coordinates. ``"zscore"`` relies on
        the provided (or inferred) mean and standard deviation, whereas
        ``"minmax"`` rescales to ``[0, 1]`` on the configured range.
    n_frequencies:
        Number of Fourier frequencies used for positional encoding.
    frequency_base:
        Base multiplier for the Fourier frequencies. Each subsequent frequency
        is ``frequency_base * 2**k`` for ``k`` in ``[0, n_frequencies)``.
    sensor_id:
        Optional SRF sensor identifier used when estimating missing FWHM values.
    default_fwhm_nm:
        Constant fallback FWHM value (in nanometres) applied when no other
        information is available. When left ``None`` and ``sensor_id`` is also
        ``None`` a uniform spacing heuristic is used.
    pooled_kind:
        Reduction strategy used for pooled token sequence.
    epsilon:
        Numerical stability constant applied to denominators.
    """

    value_norm: Literal["zscore", "robust"] = "robust"
    value_mean: np.ndarray | Sequence[float] | float | None = None
    value_std: np.ndarray | Sequence[float] | float | None = None
    lambda_norm: Literal["zscore", "minmax"] = "zscore"
    lambda_mean_nm: float | None = None
    lambda_std_nm: float | None = None
    lambda_min_nm: float | None = None
    lambda_max_nm: float | None = None
    n_frequencies: int = 8
    frequency_base: float = 1.0
    sensor_id: str | None = None
    srf_registry: SRFRegistry | None = None
    default_fwhm_nm: float | None = None
    pooled_kind: Literal["mean", "median"] = "mean"
    epsilon: float = 1e-6


@dataclass(slots=True)
class TokenMeta:
    """Metadata accompanying a tokenised spectrum."""

    axis_unit: AxisUnit
    value_norm: Literal["zscore", "robust"]
    wavelength_encoding: Literal["zscore", "minmax"]
    n_frequencies: int
    width_unit: Literal["nm"]
    used_fwhm_default: bool
    invalid_mask: np.ndarray
    pooled_kind: Literal["mean", "median"]


@dataclass(slots=True)
class Tokens:
    """Container for band-wise tokens and their pooled summary."""

    bands: np.ndarray
    pooled: np.ndarray
    meta: TokenMeta


class BandTokenizer:
    """Transform spectral bands into fixed-dimensional token representations."""

    def __init__(self, config: BandTokConfig | None = None):
        self._config = config or BandTokConfig()

    @property
    def config(self) -> BandTokConfig:
        return self._config

    def __call__(
        self,
        values: Sequence[float],
        lambda_or_nu: Sequence[float],
        *,
        axis_unit: AxisUnit,
        fwhm: Sequence[float] | float | None = None,
        config: BandTokConfig | None = None,
    ) -> Tokens:
        cfg = config or self._config

        values_arr = np.asarray(values, dtype=np.float64)
        if values_arr.ndim != 1:
            msg = "values must be a one-dimensional array"
            raise ValueError(msg)

        lambda_arr = self._to_wavelength_nm(lambda_or_nu, axis_unit)
        if lambda_arr.shape != values_arr.shape:
            msg = "Spectral axis and values must share the same length"
            raise ValueError(msg)

        fwhm_nm, used_fallback = self._prepare_fwhm(fwhm, axis_unit, lambda_arr, cfg)

        value_tokens = self._normalise_values(values_arr, cfg)
        lambda_norm = self._normalise_lambda(lambda_arr, cfg)
        fourier_feats = self._fourier_features(lambda_norm, cfg)
        width_ratio = self._bandwidth_ratio(lambda_arr, fwhm_nm, cfg)

        bands = np.concatenate(
            [
                value_tokens[:, None],
                lambda_norm[:, None],
                fourier_feats,
                width_ratio[:, None],
            ],
            axis=1,
        )

        pooled = self._pool_tokens(bands, cfg)

        meta = TokenMeta(
            axis_unit=axis_unit,
            value_norm=cfg.value_norm,
            wavelength_encoding=cfg.lambda_norm,
            n_frequencies=cfg.n_frequencies,
            width_unit="nm",
            used_fwhm_default=used_fallback,
            invalid_mask=~np.isfinite(values_arr),
            pooled_kind=cfg.pooled_kind,
        )

        return Tokens(bands=bands, pooled=pooled, meta=meta)

    def _pool_tokens(self, bands: np.ndarray, cfg: BandTokConfig) -> np.ndarray:
        if bands.size == 0:
            return np.zeros(bands.shape[1], dtype=np.float64)
        if cfg.pooled_kind == "mean":
            pooled = np.nanmean(bands, axis=0)
        else:
            pooled = np.nanmedian(bands, axis=0)
        return pooled.astype(np.float64, copy=False)

    def _normalise_values(self, values: np.ndarray, cfg: BandTokConfig) -> np.ndarray:
        finite_mask = np.isfinite(values)
        cleaned = values.copy()
        cleaned[~finite_mask] = 0.0

        eps = cfg.epsilon

        if cfg.value_norm == "zscore" and cfg.value_mean is not None and cfg.value_std is not None:
            mean = _broadcastable(cfg.value_mean, cleaned.shape)
            std = _broadcastable(cfg.value_std, cleaned.shape)
        else:
            valid = cleaned[finite_mask]
            if valid.size == 0:
                mean = 0.0
                std = 1.0
            elif cfg.value_norm == "zscore":
                mean = float(np.mean(valid))
                std = float(np.std(valid))
            else:
                median = float(np.median(valid))
                q25, q75 = np.quantile(valid, [0.25, 0.75])
                mean = median
                std = float(q75 - q25)

        std = np.maximum(np.asarray(std, dtype=np.float64), eps)
        mean = np.asarray(mean, dtype=np.float64)

        tokens = (cleaned - mean) / std
        tokens[~finite_mask] = 0.0
        return tokens

    def _normalise_lambda(self, lambda_nm: np.ndarray, cfg: BandTokConfig) -> np.ndarray:
        eps = cfg.epsilon
        if cfg.lambda_norm == "zscore":
            mean = cfg.lambda_mean_nm if cfg.lambda_mean_nm is not None else np.mean(lambda_nm)
            std = cfg.lambda_std_nm if cfg.lambda_std_nm is not None else np.std(lambda_nm)
            std = float(max(std, eps))
            normed = (lambda_nm - mean) / std
        else:
            min_nm = cfg.lambda_min_nm if cfg.lambda_min_nm is not None else float(np.min(lambda_nm))
            max_nm = cfg.lambda_max_nm if cfg.lambda_max_nm is not None else float(np.max(lambda_nm))
            span = max(max_nm - min_nm, eps)
            normed = (lambda_nm - min_nm) / span
        return np.asarray(normed, dtype=np.float64)

    def _fourier_features(self, lambda_norm: np.ndarray, cfg: BandTokConfig) -> np.ndarray:
        n_freq = int(max(cfg.n_frequencies, 0))
        if n_freq == 0:
            return np.zeros((lambda_norm.shape[0], 0), dtype=np.float64)

        freqs = cfg.frequency_base * (2.0 ** np.arange(n_freq, dtype=np.float64))
        angles = 2.0 * np.pi * np.outer(lambda_norm, freqs)
        sin_feats = np.sin(angles)
        cos_feats = np.cos(angles)
        return np.concatenate([sin_feats, cos_feats], axis=1)

    def _bandwidth_ratio(
        self, lambda_nm: np.ndarray, fwhm_nm: np.ndarray, cfg: BandTokConfig
    ) -> np.ndarray:
        deltas = _band_spacing(lambda_nm)
        eps = cfg.epsilon
        denom = np.maximum(fwhm_nm, eps)
        ratio = deltas / denom
        ratio[~np.isfinite(ratio)] = 0.0
        return ratio

    def _prepare_fwhm(
        self,
        fwhm: Sequence[float] | float | None,
        axis_unit: AxisUnit,
        lambda_nm: np.ndarray,
        cfg: BandTokConfig,
    ) -> tuple[np.ndarray, bool]:
        if fwhm is not None:
            fwhm_arr = np.asarray(fwhm, dtype=np.float64)
            if fwhm_arr.ndim == 0:
                fwhm_arr = np.full_like(lambda_nm, float(fwhm_arr))
            if fwhm_arr.shape != lambda_nm.shape:
                msg = "FWHM array must match the spectral axis length"
                raise ValueError(msg)
            if axis_unit == "cm-1":
                fwhm_arr = _wavenumber_to_wavelength_width(lambda_nm, fwhm_arr)
            return fwhm_arr, False

        fwhm_nm, source = self._estimate_fwhm(lambda_nm, cfg)
        return fwhm_nm, source != "srf"

    def _estimate_fwhm(
        self, lambda_nm: np.ndarray, cfg: BandTokConfig
    ) -> tuple[np.ndarray, str]:
        if cfg.sensor_id:
            registry = cfg.srf_registry
            srf = None
            if registry is not None:
                try:
                    srf = registry.get(cfg.sensor_id)
                except FileNotFoundError:
                    srf = None
            if srf is None:
                try:
                    srf, _ = get_srf(cfg.sensor_id)
                except ValueError:
                    srf = None
            if srf is not None:
                fwhm = _fwhm_from_srf(srf, lambda_nm)
                if fwhm is not None:
                    return fwhm, "srf"

        if cfg.default_fwhm_nm is not None:
            arr = np.full(lambda_nm.shape, float(cfg.default_fwhm_nm), dtype=np.float64)
            return arr, "default"

        deltas = _band_spacing(lambda_nm)
        mean_delta = float(np.mean(deltas)) if deltas.size else 0.0
        if mean_delta <= 0.0:
            mean_delta = 1.0
        arr = np.full(lambda_nm.shape, mean_delta, dtype=np.float64)
        return arr, "uniform"

    @staticmethod
    def _to_wavelength_nm(axis: Sequence[float], axis_unit: AxisUnit) -> np.ndarray:
        arr = np.asarray(axis, dtype=np.float64)
        if arr.ndim != 1:
            msg = "Spectral axis must be one-dimensional"
            raise ValueError(msg)
        if axis_unit == "nm":
            return arr
        if axis_unit == "cm-1":
            if np.any(arr <= 0):
                msg = "Wavenumber axis must contain positive values"
                raise ValueError(msg)
            return 1.0e7 / arr
        raise ValueError(f"Unsupported axis unit: {axis_unit!r}")


def _band_spacing(lambda_nm: np.ndarray) -> np.ndarray:
    if lambda_nm.size == 0:
        return np.empty(0, dtype=np.float64)
    diffs = np.diff(lambda_nm)
    if diffs.size == 0:
        return np.zeros_like(lambda_nm)
    spacing = np.empty_like(lambda_nm)
    spacing[1:-1] = 0.5 * (np.abs(diffs[1:]) + np.abs(diffs[:-1]))
    spacing[0] = np.abs(diffs[0])
    spacing[-1] = np.abs(diffs[-1])
    return spacing


def _wavenumber_to_wavelength_width(lambda_nm: np.ndarray, width_cm1: np.ndarray) -> np.ndarray:
    nu_cm1 = 1.0e7 / lambda_nm
    return (1.0e7 / (nu_cm1**2)) * width_cm1


def _fwhm_from_srf(srf, lambda_nm: np.ndarray) -> np.ndarray | None:
    centers = np.asarray(srf.centers_nm, dtype=np.float64)
    if centers.shape != lambda_nm.shape:
        return None

    widths: list[float] = []
    for nm, resp in zip(srf.bands_nm, srf.bands_resp, strict=True):
        nm_arr = np.asarray(nm, dtype=np.float64)
        resp_arr = np.asarray(resp, dtype=np.float64)
        if nm_arr.size == 0 or resp_arr.size == 0:
            widths.append(np.nan)
            continue
        peak = float(np.max(resp_arr))
        if not np.isfinite(peak) or peak <= 0:
            widths.append(np.nan)
            continue
        half = peak * 0.5
        try:
            left_idx = np.where(resp_arr >= half)[0][0]
            right_idx = np.where(resp_arr >= half)[0][-1]
        except IndexError:
            widths.append(np.nan)
            continue

        left_nm = _interp_half_max(nm_arr, resp_arr, left_idx, half, reverse=True)
        right_nm = _interp_half_max(nm_arr, resp_arr, right_idx, half, reverse=False)
        widths.append(abs(right_nm - left_nm))

    widths_arr = np.asarray(widths, dtype=np.float64)
    if np.all(~np.isfinite(widths_arr)):
        return None

    mean_width = float(np.nanmean(widths_arr))
    widths_arr = np.where(np.isfinite(widths_arr), widths_arr, mean_width)
    widths_arr = np.where(widths_arr <= 0, mean_width, widths_arr)
    return widths_arr


def _interp_half_max(
    nm: np.ndarray,
    resp: np.ndarray,
    idx: int,
    half: float,
    *,
    reverse: bool,
) -> float:
    i0 = idx
    i1 = idx - 1 if reverse else idx + 1
    i1 = np.clip(i1, 0, resp.size - 1)
    if i0 == i1:
        return float(nm[i0])
    x0 = resp[i0]
    x1 = resp[i1]
    y0 = nm[i0]
    y1 = nm[i1]
    if x0 == x1:
        return float(y0)
    t = (half - x0) / (x1 - x0)
    return float(y0 + t * (y1 - y0))


def _broadcastable(values: Sequence[float] | float, shape: Iterable[int]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    try:
        return np.broadcast_to(arr, shape)
    except ValueError as exc:  # pragma: no cover - defensive guard
        msg = "Provided statistics cannot be broadcast to data shape"
        raise ValueError(msg) from exc


__all__ = [
    "AxisUnit",
    "BandTokConfig",
    "BandTokenizer",
    "TokenMeta",
    "Tokens",
]

