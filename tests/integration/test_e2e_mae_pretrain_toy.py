import torch

from alchemi.models.backbone.mae import MAEBackbone


def test_toy_mae_pretraining_step():
    torch.manual_seed(0)
    model = MAEBackbone(
        embed_dim=6, depth=1, num_heads=2, decoder_dim=6, decoder_depth=1, masking_ratio=0.5
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = torch.randn(2, 8, 6)

    output = model.forward_mae(batch)
    target = batch
    loss = torch.nn.functional.mse_loss(output.decoded, target)
    loss.backward()
    optimizer.step()
    assert loss.item() >= 0.0
