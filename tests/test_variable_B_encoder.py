import torch

from alchemi.models import SetEncoder, SpectralBasisProjector


def test_basis_projector_variable_B():
    K = 64
    proj = SpectralBasisProjector(K=K)
    lam_A = torch.linspace(900, 2500, 150)
    val_A = torch.rand(150)
    m_A = torch.ones(150, dtype=torch.bool)
    lam_B = torch.linspace(1000, 2400, 60)
    val_B = torch.rand(60)
    m_B = torch.ones(60, dtype=torch.bool)
    phi_A = proj(lam_A, val_A, m_A)
    phi_B = proj(lam_B, val_B, m_B)
    assert phi_A.shape == (K,) and phi_B.shape == (K,)
    enc = SetEncoder(dim=K, depth=1, heads=2)
    z_A = enc(phi_A.unsqueeze(0), torch.ones(1, dtype=torch.bool))
    z_B = enc(phi_B.unsqueeze(0), torch.ones(1, dtype=torch.bool))
    assert z_A.shape == z_B.shape
