import torch

from alchemi.align.cycle import CycleConfig, CycleReconstructionHeads


def _synthetic_alignment(batch: int, lab_dim: int, sensor_dim: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    z_lab = torch.randn(batch, lab_dim)
    z_sensor = torch.randn(batch, lab_dim)
    lab_axis = torch.linspace(400.0, 2500.0, lab_dim)
    sensor_axis = torch.linspace(450.0, 2400.0, sensor_dim)
    lab_targets = z_lab @ torch.randn(lab_dim, lab_dim)
    sensor_targets = z_sensor @ torch.randn(lab_dim, sensor_dim)
    lab_tokens = {"values": lab_targets, "wavelengths_nm": lab_axis}
    sensor_tokens = {"radiance": sensor_targets, "wavelengths_nm": sensor_axis}
    return z_lab, z_sensor, lab_tokens, sensor_tokens


def test_cycle_continuum_and_slope_losses_reduce():
    batch, lab_dim, sensor_dim = 16, 12, 8
    config = CycleConfig(
        enabled=True,
        hidden_dim=32,
        cycle_raw=True,
        cycle_continuum=True,
        slope_reg=True,
        continuum_weight=0.5,
        slope_weight=0.1,
    )
    heads = CycleReconstructionHeads(lab_dim, sensor_dim, config)

    z_lab, z_sensor, lab_tokens, sensor_tokens = _synthetic_alignment(
        batch, lab_dim, sensor_dim
    )

    loss, init_breakdown = heads.cycle_loss(z_lab, z_sensor, lab_tokens, sensor_tokens)
    init_cont = init_breakdown["lab_cont_mse"] + init_breakdown["sensor_cont_mse"]
    init_slope = init_breakdown.get("lab_slope", torch.zeros(())) + init_breakdown.get(
        "sensor_slope", torch.zeros(())
    )

    opt = torch.optim.Adam(heads.parameters(), lr=5e-3)
    for _ in range(200):
        opt.zero_grad()
        loss, _ = heads.cycle_loss(z_lab, z_sensor, lab_tokens, sensor_tokens)
        loss.backward()
        opt.step()

    _, final_breakdown = heads.cycle_loss(z_lab, z_sensor, lab_tokens, sensor_tokens)
    final_cont = final_breakdown["lab_cont_mse"] + final_breakdown["sensor_cont_mse"]
    final_slope = final_breakdown.get("lab_slope", torch.zeros(())) + final_breakdown.get(
        "sensor_slope", torch.zeros(())
    )

    assert final_cont < init_cont
    assert torch.isfinite(final_slope)
    assert final_slope <= init_slope * 1.1
