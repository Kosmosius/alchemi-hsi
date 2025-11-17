import numpy as np
from numpy.testing import assert_allclose

from alchemi.tokens.band_tokenizer import BandTokConfig, BandTokenizer


def test_tokenizer_swir_basic_properties():
    rng = np.random.default_rng(0)
    channels = 64
    wavelengths = np.linspace(900.0, 2500.0, channels)
    values = np.sin(np.linspace(0.0, np.pi, channels)) + 0.05 * rng.standard_normal(channels)
    fwhm = np.linspace(5.0, 15.0, channels)

    config = BandTokConfig(n_frequencies=6)
    tokenizer = BandTokenizer(config)
    tokens = tokenizer(values, wavelengths, axis_unit="nm", fwhm=fwhm)

    expected_dim = 2 * config.n_frequencies + 3
    assert tokens.bands.shape == (channels, expected_dim)
    assert tokens.pooled.shape == (expected_dim,)
    assert np.all(np.isfinite(tokens.bands))
    assert np.all(np.isfinite(tokens.pooled))
    assert tokens.meta.axis_unit == "nm"
    assert tokens.meta.n_frequencies == config.n_frequencies
    assert not tokens.meta.used_fwhm_default


def test_tokenizer_wavenumber_equivalence():
    channels = 48
    wavelengths = np.linspace(7600.0, 8200.0, channels)
    wavenumbers = 1.0e7 / wavelengths
    rng = np.random.default_rng(1)
    values = 250.0 + 2.5 * rng.standard_normal(channels)
    fwhm_nm = np.full(channels, 40.0)
    fwhm_cm1 = fwhm_nm * 1.0e7 / (wavelengths**2)

    config = BandTokConfig(n_frequencies=4, value_norm="zscore")
    tokenizer = BandTokenizer(config)
    tokens_wavelength = tokenizer(values, wavelengths, axis_unit="nm", fwhm=fwhm_nm)
    tokens_wavenumber = tokenizer(values, wavenumbers, axis_unit="cm-1", fwhm=fwhm_cm1)

    assert_allclose(tokens_wavenumber.bands, tokens_wavelength.bands, atol=1e-6)


def test_missing_fwhm_triggers_default_path():
    channels = 32
    wavelengths = np.linspace(400.0, 1000.0, channels)
    values = np.linspace(0.1, 0.9, channels)
    explicit_fwhm = np.full(channels, 12.0)

    config = BandTokConfig(default_fwhm_nm=12.0, n_frequencies=2)
    tokenizer = BandTokenizer(config)

    explicit_tokens = tokenizer(values, wavelengths, axis_unit="nm", fwhm=explicit_fwhm)
    inferred_tokens = tokenizer(values, wavelengths, axis_unit="nm", fwhm=None)

    assert_allclose(inferred_tokens.bands, explicit_tokens.bands, atol=1e-8)
    assert inferred_tokens.meta.used_fwhm_default
    assert not explicit_tokens.meta.used_fwhm_default


def test_fourier_stack_matches_frequency_config():
    channels = 10
    wavelengths = np.linspace(1000.0, 1100.0, channels)
    values = np.ones(channels)
    config = BandTokConfig(n_frequencies=3)
    tokenizer = BandTokenizer(config)

    tokens = tokenizer(values, wavelengths, axis_unit="nm", fwhm=np.full(channels, 10.0))

    expected_dim = 2 * config.n_frequencies + 3
    assert tokens.bands.shape[1] == expected_dim

