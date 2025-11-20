"""Minimal demo showcasing any-sensor tokenization flow."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from alchemi.data.cube import Cube
from alchemi.srf import project_to_sensor
from alchemi.srf.synthetic import SyntheticSensorConfig, project_lab_to_synthetic
from alchemi.srf.utils import load_sensor_srf


def _lab_spectrum(highres: np.ndarray) -> NDArray[np.float64]:
    return np.asarray(
        0.5 + 0.05 * np.sin(highres / 250.0) + 0.02 * np.cos(highres / 600.0),
        dtype=np.float64,
    )


def main() -> None:
    highres = np.linspace(400.0, 2500.0, 1024)
    lab_values = _lab_spectrum(highres)

    lab_cube = Cube(
        data=lab_values.reshape(1, 1, -1),
        axis=highres,
        axis_unit="wavelength_nm",
        value_kind="radiance",
    )
    lab_tokens = lab_cube.to_tokens()

    emit_srf = load_sensor_srf("emit")
    assert emit_srf is not None
    emit_values = project_to_sensor(highres, lab_values, emit_srf.centers_nm, srf=emit_srf)
    emit_cube = Cube(
        data=emit_values.reshape(1, 1, -1),
        axis=emit_srf.centers_nm,
        axis_unit="wavelength_nm",
        value_kind="radiance",
        srf_id="emit",
    )
    emit_tokens = emit_cube.to_tokens()

    synth_cfg = SyntheticSensorConfig(
        highres_axis_nm=highres,
        n_bands=48,
        center_jitter_nm=5.0,
        fwhm_range_nm=(8.0, 16.0),
        shape="box",
        seed=1337,
    )
    synthetic = project_lab_to_synthetic(lab_values.tolist(), highres.tolist(), synth_cfg)
    synth_cube = Cube(
        data=synthetic.values.reshape(1, 1, -1),
        axis=synthetic.centers_nm,
        axis_unit="wavelength_nm",
        value_kind="radiance",
    )
    synth_tokens = synth_cube.to_tokens()

    print("Lab tokens:", lab_tokens.bands.shape, np.linalg.norm(lab_tokens.bands))
    print("EMIT tokens:", emit_tokens.bands.shape, np.linalg.norm(emit_tokens.bands))
    print("Synthetic tokens:", synth_tokens.bands.shape, np.linalg.norm(synth_tokens.bands))


if __name__ == "__main__":
    main()
