from __future__ import annotations

import numpy as np
import pytest

from alchemi.align import HyTESNoiseConfig, build_hytes_pairs
from alchemi.data.io import HYTES_WAVELENGTHS_NM
from alchemi.physics import planck
from alchemi.srf.batch_convolve import batch_convolve_lab_to_sensor
from alchemi.srf.hytes import hytes_srf_matrix


@pytest.fixture
def lab_batch() -> dict[str, np.ndarray]:
    wavelengths = np.linspace(7200.0, 12_300.0, 512, dtype=np.float64)
    centres = wavelengths[None, :]
    ramps = np.linspace(0.2, 0.8, centres.shape[-1], dtype=np.float64)
    spectra = np.vstack(
        [
            1e-5 * (1.0 + 0.05 * np.sin(centres * 1e-3)),
            9e-6 * np.exp(-((centres - 9500.0) ** 2) / (2.0 * 1200.0**2)),
            8e-6 * (0.5 + ramps**2),
        ]
    )
    return {"wavelengths_nm": wavelengths, "radiance": spectra}


def test_build_hytes_pairs_projection_matches_srf(lab_batch: dict[str, np.ndarray]) -> None:
    result = build_hytes_pairs(lab_batch, noise_cfg=HyTESNoiseConfig(enabled=False))

    srf = hytes_srf_matrix()
    expected = batch_convolve_lab_to_sensor(
        lab_batch["wavelengths_nm"], lab_batch["radiance"], srf
    )

    np.testing.assert_allclose(result["wavelengths_nm"], HYTES_WAVELENGTHS_NM)
    np.testing.assert_allclose(result["radiance"], expected, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(result["radiance_clean"], expected, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(result["bt_noise"], 0.0)


def test_bt_noise_converts_via_planck_derivative(lab_batch: dict[str, np.ndarray]) -> None:
    base = build_hytes_pairs(lab_batch, noise_cfg=HyTESNoiseConfig(enabled=False))

    rng = np.random.default_rng(1234)
    noisy = build_hytes_pairs(
        lab_batch,
        noise_cfg=HyTESNoiseConfig(enabled=True, nedt_mK=200.0, rng=rng),
    )

    delta_radiance = noisy["radiance"] - base["radiance"]
    bt_noise = noisy["bt_noise"]

    bt = planck.radiance_to_bt_K(base["radiance"], HYTES_WAVELENGTHS_NM)
    eps = 1e-4
    hi = planck.bt_K_to_radiance(bt + eps, HYTES_WAVELENGTHS_NM)
    lo = planck.bt_K_to_radiance(bt - eps, HYTES_WAVELENGTHS_NM)
    numeric_jacobian = (hi - lo) / (2.0 * eps)

    expected_delta = numeric_jacobian * bt_noise
    np.testing.assert_allclose(delta_radiance, expected_delta, rtol=5e-3, atol=1e-9)
