import torch
import torch.nn.functional as F

from alchemi.models.heads.unmix_head import LinearUnmixHead


def test_linear_unmix_head_unbatched_reconstruction() -> None:
    basis_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    head = LinearUnmixHead(embed_dim=2, k=3)
    head.fc.weight.data.zero_()
    head.fc.bias.data = torch.tensor([0.0, 1.0, 2.0])

    z = torch.tensor([0.5, -0.5])
    output = head(z, basis_emb)

    expected_frac = F.softmax(head.fc.bias, dim=-1)
    expected_recon = (expected_frac.unsqueeze(-1) * basis_emb).sum(dim=0)

    torch.testing.assert_close(output["frac"], expected_frac)
    torch.testing.assert_close(output["recon"], expected_recon)


def test_linear_unmix_head_batched_reconstruction() -> None:
    basis_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    head = LinearUnmixHead(embed_dim=2, k=3)
    head.fc.bias.data.zero_()
    head.fc.weight.data = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])

    z = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    output = head(z, basis_emb)

    logits = z @ head.fc.weight.data.T
    expected_frac = F.softmax(logits, dim=-1)
    expected_recon = (expected_frac.unsqueeze(-1) * basis_emb.unsqueeze(0)).sum(dim=1)

    assert output["frac"].shape == expected_frac.shape
    assert output["recon"].shape == expected_recon.shape
    torch.testing.assert_close(output["frac"], expected_frac)
    torch.testing.assert_close(output["recon"], expected_recon)
