import torch

from spectra.models import SpectralTokenizer


def test_tokenizer_masks_variable_B():
    tokenizer = SpectralTokenizer(
        context_size=1,
        num_pos_frequencies=2,
        include_log_lambda=True,
        include_normalized_wavelength=True,
    )

    cube = torch.zeros(2, 2, 2, 4)
    cube[0] = torch.arange(16.0).view(2, 2, 4)
    cube[1, ..., 1:] = torch.arange(12.0).view(2, 2, 3)

    wavelengths = torch.tensor(
        [
            [400.0, 500.0, 600.0, 0.0],
            [0.0, 450.0, 650.0, 850.0],
        ]
    )
    band_valid_mask = torch.tensor([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=torch.bool)

    tokens, info = tokenizer(cube, wavelengths, band_valid_mask)

    per_band_dim = tokenizer.per_band_feature_dim
    tokens_view = tokens.view(2, 4, 4, per_band_dim)

    assert tokens.shape == (2, 4, per_band_dim * 4)
    assert info["attn_mask"].shape == (2, 4)
    assert torch.equal(info["band_pad_mask"], ~band_valid_mask)

    # Padded bands should produce zeroed features.
    assert torch.allclose(tokens_view[0, :, -1], torch.zeros_like(tokens_view[0, :, -1]))
    assert torch.allclose(tokens_view[1, :, 0], torch.zeros_like(tokens_view[1, :, 0]))


def test_wavelength_permutation_changes_encoding():
    cube = torch.ones(1, 1, 1, 3)
    wavelengths = torch.tensor([500.0, 600.0, 700.0])
    valid = torch.ones(3, dtype=torch.bool)

    tokenizer = SpectralTokenizer(context_size=1, num_pos_frequencies=1)
    tokens, _ = tokenizer(cube, wavelengths, valid)

    per_band_dim = tokenizer.per_band_feature_dim
    baseline = tokens.view(1, 1, 3, per_band_dim)

    cube_perm = cube[..., [2, 0, 1]]
    wavelengths_perm = wavelengths[[2, 0, 1]]
    valid_perm = valid[[2, 0, 1]]
    tokens_perm, _ = tokenizer(cube_perm, wavelengths_perm, valid_perm)
    permuted = tokens_perm.view(1, 1, 3, per_band_dim)

    # Reordering bands should reorder the corresponding token sub-vectors.
    assert torch.allclose(permuted, baseline[..., [2, 0, 1], :])
