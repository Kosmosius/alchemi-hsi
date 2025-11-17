"""Property-based tests for Planck-law radiance/temperature conversions."""

from __future__ import annotations

import numpy as np
import pytest

try:  # pragma: no cover - optional dependency guard
    from hypothesis import given, settings
    from hypothesis import strategies as st
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    pytestmark = pytest.mark.skip(reason="Hypothesis is required for property-based tests")
else:
    from alchemi.physics import planck

    @st.composite
    def _paired_arrays(
        draw,
        *,
        min_size: int = 1,
        max_size: int = 6,
        wl_bounds: tuple[float, float],
        val_bounds: tuple[float, float],
    ):
        size = draw(st.integers(min_value=min_size, max_value=max_size))
        wl_values = draw(
            st.lists(
                st.floats(
                    min_value=wl_bounds[0],
                    max_value=wl_bounds[1],
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=size,
                max_size=size,
            )
        )
        val_values = draw(
            st.lists(
                st.floats(
                    min_value=val_bounds[0],
                    max_value=val_bounds[1],
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=size,
                max_size=size,
            )
        )
        wavelengths = np.asarray(wl_values, dtype=np.float64)
        values = np.asarray(val_values, dtype=np.float64)
        return np.sort(wavelengths), values

    @settings(max_examples=60, deadline=None)
    @given(
        _paired_arrays(
            wl_bounds=(8000.0, 14000.0),
            val_bounds=(240.0, 340.0),
        )
    )
    def test_bt_radiance_round_trip(data):
        """Radiance<->BT round-trip should recover the original temperatures."""

        wavelengths, temps = data
        radiance = planck.bt_K_to_radiance(temps, wavelengths)
        recovered = planck.radiance_to_bt_K(radiance, wavelengths)
        np.testing.assert_allclose(recovered, temps, rtol=1e-10, atol=1e-8)

    @settings(max_examples=40, deadline=None)
    @given(
        _paired_arrays(
            wl_bounds=(8000.0, 14000.0),
            val_bounds=(240.0, 340.0),
        )
    )
    def test_radiance_bt_round_trip(data):
        """BT<->radiance round-trip should recover the original radiances."""

        wavelengths, temps = data
        radiance = planck.bt_K_to_radiance(temps, wavelengths)
        recovered = planck.bt_K_to_radiance(
            planck.radiance_to_bt_K(radiance, wavelengths),
            wavelengths,
        )
        np.testing.assert_allclose(recovered, radiance, rtol=1e-10, atol=1e-8)
