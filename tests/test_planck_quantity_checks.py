import numpy as np
import pytest

from alchemi.physics.planck import radiance_to_bt
from alchemi.types import QuantityKind, Spectrum, ValueUnits, WavelengthGrid


def test_planck_rejects_reflectance():
    spectrum = Spectrum(
        wavelengths=WavelengthGrid(np.array([400.0, 500.0])),
        values=np.array([0.1, 0.2]),
        kind=QuantityKind.REFLECTANCE,
        units=ValueUnits.REFLECTANCE_FRACTION,
    )
    with pytest.raises(ValueError):
        radiance_to_bt(spectrum)

