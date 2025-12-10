from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..spectral.srf import SensorSRF, SRFProvenance
from ..srf.registry import sensor_srf_from_legacy
from ..types import SRFMatrix

_SRF_ROOT = Path("resources/srfs")


def _validate_srf_matrix(data: dict[str, Any]) -> SRFMatrix:
    try:
        sensor = data["sensor"]
        centers = np.asarray(data["centers_nm"], dtype=np.float64)
        bands = data["bands"]
    except KeyError as exc:  # pragma: no cover - defensive guard
        msg = f"Missing SRF field: {exc.args[0]}"
        raise ValueError(msg) from exc

    bands_nm = []
    bands_resp = []
    for idx, band in enumerate(bands):
        try:
            nm = np.asarray(band["nm"], dtype=np.float64)
            resp = np.asarray(band["resp"], dtype=np.float64)
        except KeyError as exc:  # pragma: no cover - defensive guard
            msg = f"Band {idx} missing field: {exc.args[0]}"
            raise ValueError(msg) from exc
        bands_nm.append(nm)
        bands_resp.append(resp)

    srf = SRFMatrix(
        sensor=sensor,
        centers_nm=centers,
        bands_nm=bands_nm,
        bands_resp=bands_resp,
        version=data.get("version", "placeholder-v0"),
        cache_key=data.get("cache_key"),
        bad_band_mask=data.get("bad_band_mask"),
        bad_band_windows_nm=data.get("bad_band_windows_nm"),
    )
    return srf.normalize_rows_trapz()


def _load_json(path: Path) -> SRFMatrix:
    data = json.loads(path.read_text())
    return _validate_srf_matrix(data)


def _load_npy(path: Path) -> SRFMatrix:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        payload = {k: arr[k].tolist() for k in arr.files}
    elif isinstance(arr, np.ndarray) and arr.shape == () and isinstance(arr.item(), dict):
        payload = arr.item()
    else:  # pragma: no cover - defensive guard
        msg = f"Unsupported NPY payload in {path}"
        raise ValueError(msg)
    return _validate_srf_matrix(payload)


def _resolve_path(sensor_id: str, base_path: Path) -> Path:
    base = sensor_id.lower()
    candidates = [
        base_path / f"{base}.json",
        base_path / f"{base}.npy",
        base_path / f"{base}.npz",
        base_path / f"{base}_srfs.json",
        base_path / f"{base}_srfs.npy",
        base_path / f"{base}_srfs.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"SRF file for {sensor_id!r} not found under {base_path}")


def get_srf(sensor_id: str, base_path: str | Path | None = None) -> SRFMatrix:
    """Load an SRF matrix for ``sensor_id``.

    The loader understands JSON payloads that mirror :class:`SRFMatrix` along
    with optional ``.npy``/``.npz`` files that contain an equivalent
    dictionary. Each band is normalized individually using trapezoidal
    integration to ensure unit area.

    For new code prefer :func:`get_sensor_srf`, which returns the canonical
    :class:`~alchemi.spectral.srf.SensorSRF` payload along with SRF metadata.
    """

    root = Path(base_path) if base_path is not None else _SRF_ROOT
    path = _resolve_path(sensor_id, root)
    if path.suffix == ".json":
        srf = _load_json(path)
    elif path.suffix in {".npy", ".npz"}:
        srf = _load_npy(path)
    else:  # pragma: no cover - defensive guard
        msg = f"Unsupported SRF extension: {path.suffix}"
        raise ValueError(msg)

    if srf.centers_nm.size == 0:
        raise ValueError(f"SRF for {sensor_id!r} contains no bands: {path}")

    return srf


def get_sensor_srf(sensor_id: str, base_path: str | Path | None = None) -> SensorSRF:
    """Load a canonical :class:`~alchemi.spectral.srf.SensorSRF` for ``sensor_id``.

    This helper wraps the legacy :class:`~alchemi.types.SRFMatrix` loader for
    compatibility with existing SRF resources while exposing the public SensorSRF
    API used by modern adapters. Per-band SRFs are normalized so that a flat
    spectrum remains flat after convolution.

    Parameters
    ----------
    sensor_id:
        Sensor identifier matching a file under ``resources/srfs``.
    base_path:
        Optional override for the SRF resource root. Defaults to
        ``resources/srfs`` within the repository.
    """

    legacy = get_srf(sensor_id, base_path=base_path)
    valid_mask = None
    if legacy.bad_band_mask is not None:
        valid_mask = ~np.asarray(legacy.bad_band_mask, dtype=bool)

    sensor_srf = sensor_srf_from_legacy(
        legacy,
        provenance=SRFProvenance.OFFICIAL,
        valid_mask=valid_mask,
    )
    meta = dict(sensor_srf.meta)
    if legacy.bad_band_windows_nm is not None:
        meta["bad_band_windows_nm"] = list(legacy.bad_band_windows_nm)
    sensor_srf.meta = meta
    return sensor_srf


def get_band_srf(sensor_id: str, band_idx: int) -> np.ndarray:
    srf = get_srf(sensor_id)
    try:
        return srf.bands_resp[band_idx]
    except IndexError as exc:  # pragma: no cover - defensive guard
        msg = f"Band index {band_idx} out of range for sensor {sensor_id!r}"
        raise IndexError(msg) from exc


__all__ = ["get_srf", "get_band_srf", "get_sensor_srf"]
