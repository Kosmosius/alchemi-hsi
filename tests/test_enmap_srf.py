from __future__ import annotations

import numpy as np

from alchemi.srf import SRFRegistry, convolve_lab_to_sensor, enmap_srf_matrix
from alchemi.types import Spectrum, SpectrumKind, WavelengthGrid


def _flat_spectrum(start: float, stop: float, value: float = 0.32) -> Spectrum:
    wavelengths = np.linspace(start, stop, 4096, dtype=np.float64)
    return Spectrum(
        WavelengthGrid(wavelengths),
        np.full_like(wavelengths, value),
        SpectrumKind.REFLECTANCE,
        "unitless",
    )


def test_srf_integral_enmap(tmp_path):
    srf = enmap_srf_matrix(cache_dir=tmp_path)
    integrals = srf.row_integrals()
    assert np.allclose(integrals, 1.0, atol=1e-6)

    vnir = srf.centers_nm[srf.centers_nm <= 999.0]
    swir = srf.centers_nm[srf.centers_nm > 999.0]
    assert vnir.size > 0 and swir.size > 0
    assert abs(vnir.max() - swir.min()) <= 5.0


def test_srf_flat_response_enmap(tmp_path):
    srf = enmap_srf_matrix(cache_dir=tmp_path, force=True)
    spec = _flat_spectrum(srf.bands_nm[0][0], srf.bands_nm[-1][-1])
    convolved = convolve_lab_to_sensor(spec, srf)
    assert np.allclose(convolved.values, spec.values[0], atol=1e-6)
    assert np.isclose(convolved.values.mean(), spec.values[0], atol=1e-6)


def test_srf_cache_roundtrip_enmap(tmp_path):
    srf = enmap_srf_matrix(cache_dir=tmp_path)
    cache_path = tmp_path / "enmap.json"
    assert cache_path.exists()

    registry = SRFRegistry(tmp_path)
    cached = registry.get("enmap")

    np.testing.assert_allclose(cached.centers_nm, srf.centers_nm)
    for nm_a, nm_b in zip(cached.bands_nm, srf.bands_nm, strict=True):
        np.testing.assert_allclose(nm_a, nm_b)
    for resp_a, resp_b in zip(cached.bands_resp, srf.bands_resp, strict=True):
        np.testing.assert_allclose(resp_a, resp_b)
    assert cached.cache_key == srf.cache_key
