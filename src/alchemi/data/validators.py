from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def validate_dataset(cfg: dict[str, Any]) -> None:
    dp = cfg.get("data", {})
    for key in ["paths", "wavelengths"]:
        if key not in dp:
            raise ValueError(f"Missing data.{key} in config")


def validate_srf_dir(srf_root: str | os.PathLike[str]) -> None:
    root = Path(srf_root)
    if not root.exists():
        raise FileNotFoundError(f"SRF root not found: {root}")
    for p in root.glob("*.json"):
        obj = json.loads(p.read_text())
        centers = np.array(obj["centers_nm"], dtype=float)
        if not np.all(np.diff(centers) > 0):
            raise ValueError(f"Non-monotonic centers in {p.name}")
        for b in obj["bands"]:
            nm = np.array(b["nm"], dtype=float)
            resp = np.array(b["resp"], dtype=float)
            if nm.shape != resp.shape or (resp < 0).any():
                raise ValueError(f"Invalid SRF row in {p.name}")
