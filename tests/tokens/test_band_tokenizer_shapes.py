import numpy as np

from alchemi.tokens import BandTokConfig, BandTokenizer


def test_swir_tokens_shape_and_stability():
    rng = np.random.default_rng(0)
    channels = 73
    axis = np.sort(rng.uniform(420.0, 2450.0, size=channels))
    values = rng.normal(loc=0.2, scale=0.05, size=channels)
    width = rng.uniform(5.0, 15.0, size=channels)

    config = BandTokConfig(n_fourier_frequencies=4, token_dim=48)
    tokenizer = BandTokenizer(config)
    tokens = tokenizer(values, axis, axis_unit="nm", width=width)

    assert tokens.bands.shape == (channels, config.token_dim)
    assert tokens.meta.token_dim == config.token_dim
    assert np.all(np.isfinite(tokens.bands))

    perm = rng.permutation(channels)
    tokens_shuffled = tokenizer(values[perm], axis[perm], axis_unit="nm", width=width[perm])
    assert tokens_shuffled.bands.shape == tokens.bands.shape
    assert not np.allclose(tokens_shuffled.bands, tokens.bands)


def test_lwir_tokens_support_wavenumber_axes():
    rng = np.random.default_rng(2)
    channels = 48
    axis_cm1 = np.sort(rng.uniform(800.0, 1300.0, size=channels))
    values = 250.0 + rng.normal(scale=1.0, size=channels)
    width_cm1 = rng.uniform(1.0, 4.0, size=channels)

    config = BandTokConfig(n_fourier_frequencies=3, lambda_unit="cm-1", token_dim=40)
    tokenizer = BandTokenizer(config)
    tokens = tokenizer(values, axis_cm1, axis_unit="cm-1", width=width_cm1)

    assert tokens.bands.shape == (channels, config.token_dim)
    assert np.all(np.isfinite(tokens.bands))
    assert np.all(np.isfinite(tokens.pooled))

    perm = rng.permutation(channels)
    shuffled = tokenizer(values[perm], axis_cm1[perm], axis_unit="cm-1", width=width_cm1[perm])
    assert shuffled.bands.shape == tokens.bands.shape
    assert not np.allclose(shuffled.bands, tokens.bands)


def test_unknown_sensor_width_defaults_are_inferred():
    rng = np.random.default_rng(8)
    channels = 12
    axis = np.sort(rng.uniform(400.0, 2500.0, size=channels))
    values = rng.normal(size=channels)

    config = BandTokConfig(include_width=True, sensor_id=None, token_dim=24)
    tokenizer = BandTokenizer(config)
    tokens = tokenizer(values, axis, axis_unit="nm", width=None)

    assert tokens.bands.shape == (channels, config.token_dim)
    assert np.all(np.isfinite(tokens.bands))
    assert tokens.meta.used_default_width is True
