import json

import numpy as np

from alchemi.data.pairing import LabSensorCache
from alchemi.srf.registry import SRFRegistry


def test_pairing_cache(tmp_path):
    srf_dir = tmp_path / "srf"
    srf_dir.mkdir()
    obj = {
        "sensor": "foo",
        "version": "v1",
        "centers_nm": [1000.0],
        "bands": [{"nm": [990, 1000, 1010], "resp": [0, 1, 0]}],
    }
    (srf_dir / "foo.json").write_text(json.dumps(obj))
    reg = SRFRegistry(srf_dir)

    nm = np.linspace(950, 1050, 51)
    vals = np.stack([np.sin(nm / 1000.0), np.cos(nm / 1000.0)], axis=0)
    cache = LabSensorCache(tmp_path / "cache")
    out1 = cache.convolve(nm, vals, "foo", reg)
    out2 = cache.convolve(nm, vals, "foo", reg)
    assert np.allclose(out1, out2)
