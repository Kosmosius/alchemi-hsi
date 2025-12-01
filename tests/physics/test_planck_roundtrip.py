import numpy as np

from alchemi.physics.planck import bt_K_to_radiance, radiance_to_bt_K


def test_planck_roundtrip_within_tolerance():
    wavelengths = np.linspace(800.0, 1200.0, 4)
    temps = np.array([280.0, 300.0, 320.0, 340.0])
    radiance = bt_K_to_radiance(temps, wavelengths)
    recovered = radiance_to_bt_K(radiance, wavelengths)
    assert np.allclose(recovered, temps, atol=0.1)
