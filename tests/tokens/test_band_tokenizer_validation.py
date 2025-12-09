import numpy as np
import pytest

from alchemi.tokens import BandTokConfig, BandTokenizer


def test_tokenizer_outputs_shapes_with_optional_features():
    channels = 10
    axis = np.linspace(400.0, 800.0, channels)
    values = np.linspace(0.1, 1.0, channels)
    width = np.full(channels, 6.0)
    srf = np.ones((channels, 6))

    cfg = BandTokConfig(
        include_width=True,
        include_srf_embed=True,
        srf_embed_dim=4,
        token_dim=16,
        n_fourier_frequencies=2,
    )
    tokenizer = BandTokenizer(cfg)

    tokens = tokenizer(values, axis, axis_unit="nm", width=width, srf_row=srf)

    assert tokens.bands.shape == (channels, cfg.token_dim)
    assert tokens.pooled.shape == (cfg.token_dim,)
    assert np.all(np.isfinite(tokens.bands))
    assert np.all(np.isfinite(tokens.pooled))


def test_length_mismatch_raises_helpful_error():
    cfg = BandTokConfig(include_width=False)
    tokenizer = BandTokenizer(cfg)

    values = np.ones(5)
    axis = np.linspace(400.0, 600.0, 4)

    with pytest.raises(ValueError, match="matching lengths"):
        tokenizer(values, axis, axis_unit="nm")


def test_width_shape_mismatch_raises_error():
    cfg = BandTokConfig()
    tokenizer = BandTokenizer(cfg)

    values = np.ones(6)
    axis = np.linspace(500.0, 700.0, 6)
    width = np.ones(4)

    with pytest.raises(ValueError, match="width must align"):
        tokenizer(values, axis, axis_unit="nm", width=width)


def test_srf_shape_mismatch_raises_error():
    cfg = BandTokConfig(include_srf_embed=True, include_width=False)
    tokenizer = BandTokenizer(cfg)

    values = np.ones(4)
    axis = np.linspace(500.0, 520.0, 4)
    srf = np.ones((3, 2))

    with pytest.raises(ValueError, match="srf_row must align"):
        tokenizer(values, axis, axis_unit="nm", srf_row=srf)


def test_global_stats_required_for_global_zscore():
    cfg = BandTokConfig(value_norm="global_zscore", include_width=False)
    tokenizer = BandTokenizer(cfg)

    values = np.ones(3)
    axis = np.linspace(600.0, 620.0, 3)

    with pytest.raises(ValueError, match="ValueStats"):
        tokenizer(values, axis, axis_unit="nm")
