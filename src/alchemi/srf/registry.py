from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..types import SRFMatrix
from ..utils.logging import get_logger

_LOG = get_logger(__name__)


class SRFRegistry:
    def __init__(self, root: str | Path = "data/srf"):
        self.root = Path(root)
        self._cache: dict[str, SRFMatrix] = {}

    def _hash(self, payload: str) -> str:
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def get(self, sensor: str) -> SRFMatrix:
        k = sensor.lower()
        if k in self._cache:
            return self._cache[k]
        path = self.root / f"{k}.json"
        if not path.exists():
            raise FileNotFoundError(f"SRF file not found: {path}")
        obj = json.loads(path.read_text())
        centers = np.array(obj["centers_nm"], dtype=np.float64)
        bands_nm = [np.array(b["nm"], dtype=np.float64) for b in obj["bands"]]
        bands_resp = [np.array(b["resp"], dtype=np.float64) for b in obj["bands"]]
        srf = SRFMatrix(
            obj["sensor"],
            centers,
            bands_nm,
            bands_resp,
            version=obj.get("version", "v1"),
        )
        srf = srf.normalize_trapz()
        srf.cache_key = self._hash(path.read_text()[:2048])
        self._cache[k] = srf
        _LOG.info("Loaded SRF for %s (%d bands)", k, len(centers))
        return srf
