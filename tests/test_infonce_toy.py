import torch

from alchemi.losses import InfoNCELoss


def test_infonce_aligns_toy():
    torch.manual_seed(0)
    z = torch.randn(8, 16)
    z_pos = z + 0.01 * torch.randn_like(z)
    loss = InfoNCELoss()(z, z_pos)
    assert loss.item() < 3.0
