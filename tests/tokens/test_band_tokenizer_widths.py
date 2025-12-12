import numpy as np

from alchemi.srf.utils import load_sensor_srf, resolve_band_widths
from alchemi.tokens import BandTokConfig, BandTokenizer


def _width_feature_column(tokenizer: BandTokenizer, values: np.ndarray, axis: np.ndarray):
    tokens = tokenizer(values, axis, axis_unit="nm", width=None)
    return tokens.bands[:, -1], tokens.meta.used_default_width


def test_band_tokenizer_uses_srf_widths_for_known_sensors():
    sensors = ["emit", "enmap", "avirisng", "hytes"]
    for sensor in sensors:
        sensor_srf = load_sensor_srf(sensor)
        assert sensor_srf is not None, f"missing SRF for {sensor}"
        centers = np.asarray(sensor_srf.centers_nm, dtype=float)

        cfg = BandTokConfig(
            n_fourier_frequencies=0,
            value_norm="none",
            include_width=True,
            sensor_id=sensor,
        )
        tokenizer = BandTokenizer(cfg)
        values = np.ones_like(centers)

        expected_widths, _, _ = resolve_band_widths(sensor, centers, srf=sensor_srf)
        expected_normalized = tokenizer._normalise_width(expected_widths)

        width_column, used_default = _width_feature_column(tokenizer, values, centers)

        assert not used_default
        assert np.allclose(width_column, expected_normalized)


def test_band_tokenizer_respects_fwhm_when_no_srf():
    axis = np.array([400.0, 500.0, 600.0])
    fwhm = np.array([6.0, 7.0, 8.0])
    cfg = BandTokConfig(n_fourier_frequencies=0, value_norm="none", include_width=True, sensor_id=None)
    tokenizer = BandTokenizer(cfg)

    tokens = tokenizer(np.ones_like(axis), axis, axis_unit="nm", width=fwhm)

    expected = tokenizer._normalise_width(fwhm)
    assert np.allclose(tokens.bands[:, -1], expected)
    assert not tokens.meta.used_default_width


def test_band_tokenizer_flags_default_widths_when_missing():
    axis = np.array([100.0, 200.0, 400.0])
    cfg = BandTokConfig(n_fourier_frequencies=0, value_norm="none", include_width=True, sensor_id=None)
    tokenizer = BandTokenizer(cfg)

    tokens = tokenizer(np.ones_like(axis), axis, axis_unit="nm")

    assert tokens.meta.used_default_width
