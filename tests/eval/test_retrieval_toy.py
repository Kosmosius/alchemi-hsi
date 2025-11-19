import torch

from alchemi.align.losses import info_nce_symmetric
from alchemi.eval.retrieval import compute_retrieval_at_k, random_retrieval_at_k


def test_toy_alignment_training_beats_random():
    torch.manual_seed(0)
    batch, spec_dim, embed_dim = 16, 12, 8
    lab = torch.randn(batch, spec_dim)
    transform = torch.randn(spec_dim, spec_dim)
    sensor = lab @ transform + 0.05 * torch.randn(batch, spec_dim)

    encoder_lab = torch.nn.Linear(spec_dim, embed_dim, bias=False)
    encoder_sensor = torch.nn.Linear(spec_dim, embed_dim, bias=False)
    opt = torch.optim.Adam(
        list(encoder_lab.parameters()) + list(encoder_sensor.parameters()),
        lr=1e-2,
    )

    for _ in range(300):
        opt.zero_grad()
        z_lab = encoder_lab(lab)
        z_sensor = encoder_sensor(sensor)
        loss_out = info_nce_symmetric(z_lab, z_sensor, gather_ddp=False)
        loss_out.loss.backward()
        opt.step()

    z_lab_eval = encoder_lab(lab).detach().numpy()
    z_sensor_eval = encoder_sensor(sensor).detach().numpy()
    metrics = compute_retrieval_at_k(z_lab_eval, z_sensor_eval, k=1)
    baseline = random_retrieval_at_k(batch, batch, k=1)

    assert metrics.recall >= 5 * baseline
