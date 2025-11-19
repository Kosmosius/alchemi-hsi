import torch

from alchemi.align.losses import InfoNCELossOut, info_nce_symmetric
from alchemi.losses import InfoNCELoss


def test_info_nce_symmetric_decreases_on_aligned_pairs():
    torch.manual_seed(0)
    batch, dim = 6, 8
    z_lab = torch.nn.Parameter(torch.randn(batch, dim))
    z_sensor = torch.nn.Parameter(torch.randn(batch, dim))

    optim = torch.optim.SGD([z_lab, z_sensor], lr=0.3)

    with torch.no_grad():
        initial_loss = info_nce_symmetric(
            z_lab.detach(),
            z_sensor.detach(),
            gather_ddp=False,
        ).loss.item()

    for _ in range(80):
        optim.zero_grad(set_to_none=True)
        loss_out = info_nce_symmetric(z_lab, z_sensor, gather_ddp=False)
        assert isinstance(loss_out, InfoNCELossOut)
        loss_out.loss.backward()
        optim.step()

    with torch.no_grad():
        final_loss = info_nce_symmetric(z_lab, z_sensor, gather_ddp=False).loss.item()

    assert final_loss < initial_loss - 0.2


def test_info_nce_symmetric_matches_legacy_infonce():
    torch.manual_seed(123)
    z_lab = torch.randn(4, 5)
    z_sensor = torch.randn(4, 5)

    legacy = InfoNCELoss(temperature=0.07)(z_lab, z_sensor)
    symmetric = info_nce_symmetric(
        z_lab,
        z_sensor,
        tau_init=0.07,
        learnable_tau=False,
        gather_ddp=False,
    ).loss

    torch.testing.assert_close(legacy, symmetric)


def test_info_nce_symmetric_tau_positive_and_grad():
    torch.manual_seed(1)
    z_lab = torch.randn(3, 4, requires_grad=True)
    z_sensor = torch.randn(3, 4, requires_grad=True)

    loss_out = info_nce_symmetric(z_lab, z_sensor, gather_ddp=False)
    loss_out.loss.backward()

    tau_metric = loss_out.metrics["tau"]
    assert tau_metric.item() > 0.0

    params = loss_out.parameters()
    if params:
        log_tau = params[0]
        assert isinstance(log_tau, torch.nn.Parameter)
        assert log_tau.grad is not None


def test_info_nce_symmetric_calls_gather(monkeypatch):
    calls = {"count": 0}

    def _fake_gather(x):
        calls["count"] += 1
        return x

    monkeypatch.setattr("alchemi.align.losses._ddp_is_initialized", lambda: True)
    monkeypatch.setattr("alchemi.align.losses._gather_embeddings", _fake_gather)
    monkeypatch.setattr(
        "alchemi.align.losses.dist",
        type(
            "_DummyDist",
            (),
            {"get_world_size": staticmethod(lambda: 1), "get_rank": staticmethod(lambda: 0)},
        ),
    )

    z_lab = torch.randn(2, 3)
    z_sensor = torch.randn(2, 3)
    info_nce_symmetric(z_lab, z_sensor, gather_ddp=True)

    assert calls["count"] >= 2
