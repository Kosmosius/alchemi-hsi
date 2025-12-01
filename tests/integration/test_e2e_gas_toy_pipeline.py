import torch

from alchemi.models.heads.gas_head import GasHead


def test_toy_gas_pipeline_forward_backward():
    torch.manual_seed(0)
    head = GasHead(embed_dim=4)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)
    features = torch.randn(3, 2, 4)
    targets = torch.tensor([[0.1, 0.2], [0.0, 0.3], [0.4, 0.5]])

    preds = head(features)
    loss = torch.nn.functional.mse_loss(preds, targets)
    loss.backward()
    optimizer.step()
    assert preds.shape == targets.shape
