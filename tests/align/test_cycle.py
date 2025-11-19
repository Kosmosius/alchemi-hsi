import torch

from alchemi.align.cycle import CycleAlignment, CycleConfig, CycleReconstructionHeads


def _synthetic_batch(batch: int, lab_dim: int, sensor_dim: int) -> tuple[torch.Tensor, ...]:
    transform_lab_to_sensor = torch.randn(lab_dim, sensor_dim)
    transform_sensor_to_lab = torch.randn(sensor_dim, lab_dim)

    z_lab = torch.randn(batch, lab_dim)
    sensor_tokens = z_lab @ transform_lab_to_sensor
    # Sensor latents are a noisy projection of the sensor tokens.
    z_sensor = sensor_tokens @ transform_sensor_to_lab + 0.01 * torch.randn(batch, lab_dim)
    lab_tokens = z_lab.clone()
    return z_lab, z_sensor, lab_tokens, sensor_tokens


def test_cycle_loss_reduces_sam_angle():
    batch, lab_dim, sensor_dim = 64, 6, 5
    torch.manual_seed(0)
    config = CycleConfig(
        enabled=True,
        hidden_dim=32,
        sam_weight=1.0,
        l2_weight=1.0,
        cycle_raw=True,
    )
    heads = CycleReconstructionHeads(lab_dim, sensor_dim, config)

    z_lab, z_sensor, lab_tokens, sensor_tokens = _synthetic_batch(batch, lab_dim, sensor_dim)

    _, breakdown = heads.cycle_loss(z_lab, z_sensor, lab_tokens, sensor_tokens)
    initial_sam = breakdown["sensor_sam"].item()

    optimizer = torch.optim.Adam(heads.parameters(), lr=5e-2)

    for _ in range(200):
        optimizer.zero_grad()
        loss, _ = heads.cycle_loss(z_lab, z_sensor, lab_tokens, sensor_tokens)
        loss.backward()
        optimizer.step()

    _, breakdown = heads.cycle_loss(z_lab, z_sensor, lab_tokens, sensor_tokens)
    final_sam = breakdown["sensor_sam"].item()

    assert final_sam < initial_sam


def test_cycle_alignment_keeps_infonce_finite():
    batch, lab_dim, sensor_dim = 8, 4, 3
    torch.manual_seed(0)
    z_lab, z_sensor, lab_tokens, sensor_tokens = _synthetic_batch(batch, lab_dim, sensor_dim)

    aligner = CycleAlignment(
        lab_dim,
        sensor_dim,
        CycleConfig(enabled=True, cycle_raw=True),
    )
    losses = aligner(z_lab, z_sensor, lab_tokens, sensor_tokens)

    assert torch.isfinite(losses["infonce"]) and torch.isfinite(losses["cycle"])


def test_cycle_disabled_returns_only_infonce():
    batch, lab_dim, sensor_dim = 8, 4, 3
    torch.manual_seed(0)
    z_lab, z_sensor, lab_tokens, sensor_tokens = _synthetic_batch(batch, lab_dim, sensor_dim)

    aligner = CycleAlignment(lab_dim, sensor_dim, CycleConfig(enabled=False))
    losses = aligner(z_lab, z_sensor, lab_tokens, sensor_tokens)

    assert set(losses) == {"infonce"}
    assert torch.isfinite(losses["infonce"])
