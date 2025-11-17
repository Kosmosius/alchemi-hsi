from __future__ import annotations

import math

import numpy as np
import torch

from alchemi.heads import BandDefinition, BandDepthHead
from alchemi.physics import continuum_remove


def _gaussian_spectrum(wavelengths: torch.Tensor, centers: list[float]) -> torch.Tensor:
    spectrum = torch.ones_like(wavelengths)
    for center in centers:
        depth = torch.empty(1).uniform_(0.05, 0.35).item()
        width = torch.empty(1).uniform_(10.0, 25.0).item()
        spectrum -= depth * torch.exp(-0.5 * ((wavelengths - center) / width) ** 2)
    return spectrum.clamp_min(0.02)


def _true_band_depths(
    wavelengths: torch.Tensor, spectrum: torch.Tensor, bands: list[BandDefinition]
) -> torch.Tensor:
    wl = wavelengths.cpu().numpy()
    refl = spectrum.cpu().numpy()
    values = []
    for spec in bands:
        _cont, removed = continuum_remove(wl, refl, spec.left_nm, spec.right_nm)
        idx = np.searchsorted(wl, spec.center_nm)
        idx = min(max(idx, 0), wl.size - 1)
        values.append(1.0 - removed[idx])
    return torch.tensor(values, dtype=spectrum.dtype)


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    diff = pred - target
    ss_res = torch.sum(diff * diff)
    mean_target = torch.mean(target)
    ss_tot = torch.sum((target - mean_target) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def test_banddepth_head_learns_gaussian_band_depths() -> None:
    torch.manual_seed(0)
    wavelengths = torch.linspace(900.0, 2500.0, steps=256)
    bands = [
        BandDefinition(center_nm=1050.0, left_nm=1000.0, right_nm=1100.0),
        BandDefinition(center_nm=1350.0, left_nm=1300.0, right_nm=1400.0),
        BandDefinition(center_nm=1950.0, left_nm=1880.0, right_nm=2020.0),
    ]
    head = BandDepthHead(embed_dim=16, bands=bands, hidden_dim=32, loss="l2")

    mixing = torch.randn(len(bands), 16)
    spectra: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    truths: list[torch.Tensor] = []
    for _ in range(96):
        n_abs = torch.randint(1, 3, (1,)).item()
        centers = torch.tensor([bands[i].center_nm for i in torch.randint(0, len(bands), (n_abs,))])
        spec = _gaussian_spectrum(wavelengths, centers.tolist())
        true_depths = _true_band_depths(wavelengths, spec, bands)
        spectra.append(spec)
        truths.append(true_depths)
        noisy_embed = true_depths @ mixing + 0.02 * torch.randn(16)
        features.append(noisy_embed)
    spectra = torch.stack(spectra)
    features = torch.stack(features)
    truths = torch.stack(truths)

    with torch.no_grad():
        torch.testing.assert_close(
            head.compute_targets(wavelengths, spectra), truths, atol=5e-3, rtol=1e-3
        )

    optimiser = torch.optim.Adam(head.parameters(), lr=5e-3)
    for _ in range(300):
        preds = head(features)
        target_depths = head.compute_targets(wavelengths, spectra)
        loss = head.loss(preds, target_depths)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        preds = head(features)
        target_depths = head.compute_targets(wavelengths, spectra)
        score = r2_score(preds, target_depths)
    assert not math.isnan(score)
    assert score > 0.9
