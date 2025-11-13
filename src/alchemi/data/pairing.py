from __future__ import annotations

from pathlib import Path

import numpy as np
from joblib import Memory

from ..srf import SRFRegistry, batch_convolve_lab_to_sensor
from ..types import Sample, Spectrum, SpectrumKind, WavelengthGrid


class LabSensorCache:
    def __init__(self, cache_dir: str | Path = "data/cache"):
        self.mem = Memory(location=str(cache_dir), verbose=0)

    def convolve(
        self, lab_nm: np.ndarray, lab_values: np.ndarray, sensor: str, srf_reg: SRFRegistry
    ) -> np.ndarray:
        srf = srf_reg.get(sensor)

        @self.mem.cache
        def _conv(
            lab_nm_hash: str, lab_values: np.ndarray, sensor_key: str, srf_data_hash: str
        ):
            return batch_convolve_lab_to_sensor(lab_nm, lab_values, srf)

        key_nm = str(hash(lab_nm.tobytes()))
        key_srf = srf.cache_key or "nosha"
        return _conv(key_nm, lab_values, sensor, key_srf)


class PairBuilder:
    def __init__(self, srf_root: str = "data/srf", cache_dir: str = "data/cache"):
        self.reg = SRFRegistry(srf_root)
        self.cache = LabSensorCache(cache_dir)

    def make_pairs(
        self, sensor: str, field: list[Sample], lab_samples: list[Sample], seed: int = 42
    ) -> tuple[list[Sample], list[Sample]]:
        assert len(lab_samples) >= len(field)
        lab_nm = lab_samples[0].spectrum.wavelengths.nm
        lab_vals = np.stack([s.spectrum.values for s in lab_samples], axis=0)
        conv = self.cache.convolve(lab_nm, lab_vals, sensor, self.reg)
        centers = self.reg.get(sensor).centers_nm
        lab_conv = []
        for i in range(len(field)):
            vals = conv[i]
            spec = Spectrum(
                WavelengthGrid(centers),
                vals,
                SpectrumKind.REFLECTANCE,
                "unitless",
                None,
                {"sensor": sensor, "source": "lab_convolved"},
            )
            lab_conv.append(Sample(spec, lab_samples[i].meta))
        return field, lab_conv
