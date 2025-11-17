import numpy as np

from alchemi.tokens import BandTokConfig, BandTokenizer


def test_srf_embedding_changes_token_values():
    rng = np.random.default_rng(4)
    channels = 32
    axis = np.linspace(500.0, 900.0, channels)
    values = rng.normal(size=channels)
    width = np.full(channels, 8.0)
    srf_features = rng.normal(size=(channels, 6))

    base_cfg = BandTokConfig(n_fourier_frequencies=2, token_dim=0)
    cfg_with = BandTokConfig(
        n_fourier_frequencies=2,
        include_srf_embed=True,
        srf_embed_dim=4,
        token_dim=0,
    )

    tokenizer_with = BandTokenizer(cfg_with)
    tokenizer_without = BandTokenizer(base_cfg)

    tokens_with = tokenizer_with(values, axis, axis_unit="nm", width=width, srf_row=srf_features)
    tokens_without = tokenizer_without(values, axis, axis_unit="nm", width=width)

    assert tokens_with.bands.shape[1] > tokens_without.bands.shape[1]
    srf_block = tokens_with.bands[:, -cfg_with.srf_embed_dim :]
    assert srf_block.shape[1] == cfg_with.srf_embed_dim
    assert not np.allclose(srf_block, 0.0)
    assert not np.allclose(srf_block[0], srf_block[-1])


def test_disabling_srf_embed_preserves_shapes_when_projected():
    rng = np.random.default_rng(7)
    channels = 24
    axis = np.linspace(400.0, 1000.0, channels)
    values = rng.standard_normal(channels)
    width = np.linspace(5.0, 10.0, channels)
    srf = rng.standard_normal((channels, 12))

    cfg = BandTokConfig(token_dim=48, include_srf_embed=True, srf_embed_dim=8)
    cfg_no = BandTokConfig(token_dim=48, include_srf_embed=False)

    tok = BandTokenizer(cfg)
    tok_no = BandTokenizer(cfg_no)

    tokens = tok(values, axis, axis_unit="nm", width=width, srf_row=srf)
    tokens_no = tok_no(values, axis, axis_unit="nm", width=width)

    assert tokens.bands.shape == tokens_no.bands.shape
    assert not np.allclose(tokens.bands, tokens_no.bands)
