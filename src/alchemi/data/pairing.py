from __future__ import annotations

from pathlib import Path

import numpy as np
from joblib import Memory
from numpy.typing import NDArray

from ..physics import resampling
from ..registry import srfs
from ..types import Sample, SampleMeta, Spectrum, SpectrumKind, WavelengthGrid


class LabSensorCache:
    def __init__(self, cache_dir: str | Path = "data/cache"):
        self.mem = Memory(location=str(cache_dir), verbose=0)

    def convolve(
        self, lab_nm: np.ndarray, lab_values: np.ndarray, sensor: str, srf_root: str | Path | None
    ) -> NDArray[np.float64]:
        srf = srfs.get_srf(sensor, base_path=srf_root)
        if srf is None:
            msg = f"No SRF available for sensor_id={sensor!r}"
            raise KeyError(msg)

        @self.mem.cache  # type: ignore[misc]
        def _conv(
            lab_nm_hash: str,
            lab_values: np.ndarray,
            sensor_key: str,
            srf_data_hash: str,
        ) -> NDArray[np.float64]:
            band_values: list[np.ndarray] = []
            for row in lab_values:
                spectrum = Spectrum(
                    wavelength_nm=lab_nm,
                    values=np.asarray(row, dtype=np.float64),
                    kind=SpectrumKind.REFLECTANCE,
                    units="unitless",
                )
                convolved = resampling.convolve_to_bands(spectrum, srf)
                band_values.append(np.asarray(convolved.values, dtype=np.float64))
            return np.vstack(band_values)

        key_nm = str(hash(lab_nm.tobytes()))
        key_srf = getattr(srf, "cache_key", None) or "nosha"
        return _conv(key_nm, lab_values, sensor, key_srf)


class PairBuilder:
    def __init__(self, srf_root: str = "data/srf", cache_dir: str = "data/cache"):
        self.srf_root = srf_root
        self.cache = LabSensorCache(cache_dir)

    def make_pairs(
        self, sensor: str, field: list[Sample], lab_samples: list[Sample], seed: int = 42
    ) -> tuple[list[Sample], list[Sample]]:
        assert len(lab_samples) >= len(field)
        lab_nm = lab_samples[0].spectrum.wavelength_nm
        lab_vals = np.stack([s.spectrum.values for s in lab_samples], axis=0)
        conv = self.cache.convolve(lab_nm, lab_vals, sensor, self.srf_root)
        centers = srfs.get_srf(sensor, base_path=self.srf_root).centers_nm
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
            meta = SampleMeta.from_sample(lab_samples[i])
            lab_conv.append(meta.to_sample(spec))
        return field, lab_conv
