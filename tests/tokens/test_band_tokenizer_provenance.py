import numpy as np

from alchemi.tokens import BandTokConfig, BandTokenizer


def test_provenance_embeddings_appended_when_enabled():
    channels = 16
    axis = np.linspace(400.0, 900.0, channels)
    values = np.linspace(0.0, 1.0, channels)
    provenance = np.array(["official", "gaussian"] * (channels // 2), dtype=object)

    cfg = BandTokConfig(
        n_fourier_frequencies=1,
        include_width=False,
        include_srf_provenance=True,
        srf_provenance_embed_dim=2,
        token_dim=0,
    )
    cfg_base = BandTokConfig(n_fourier_frequencies=1, include_width=False, include_srf_provenance=False)

    tokenizer = BandTokenizer(cfg)
    baseline = BandTokenizer(cfg_base)

    tokens_with = tokenizer(values, axis, axis_unit="nm", srf_provenance=provenance)
    tokens_base = baseline(values, axis, axis_unit="nm")

    assert tokens_with.bands.shape[1] > tokens_base.bands.shape[1]
    prov_block = tokens_with.bands[:, -cfg.srf_provenance_embed_dim :]
    assert prov_block.shape[1] == cfg.srf_provenance_embed_dim
    assert not np.allclose(prov_block[0], prov_block[-1])


def test_missing_provenance_skips_embeddings():
    channels = 12
    axis = np.linspace(500.0, 800.0, channels)
    values = np.cos(axis / 100.0)

    cfg = BandTokConfig(n_fourier_frequencies=1, include_width=False, include_srf_provenance=True)
    cfg_off = BandTokConfig(n_fourier_frequencies=1, include_width=False, include_srf_provenance=False)

    tokenizer = BandTokenizer(cfg)
    baseline = BandTokenizer(cfg_off)

    tokens_with = tokenizer(values, axis, axis_unit="nm")
    tokens_base = baseline(values, axis, axis_unit="nm")

    assert tokens_with.bands.shape == tokens_base.bands.shape
