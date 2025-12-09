from dataclasses import asdict

import numpy as np

from alchemi.spectral import Sample, Spectrum, ViewingGeometry


def test_sample_roundtrip_through_chip_preserves_metadata():
    wavelengths = np.array([400.0, 500.0, 600.0])
    values = np.array([0.1, 0.2, 0.3])
    viewing = ViewingGeometry(10.0, 20.0, 30.0, 40.0, 1.0)
    band_meta = {
        "center_nm": wavelengths,
        "width_nm": np.array([10.0, 10.0, 10.0]),
        "srf_source": np.array(["catalog"] * 3),
        "valid_mask": np.array([True, True, True]),
    }
    sample = Sample(
        spectrum=Spectrum(wavelength_nm=wavelengths, values=values, kind="radiance"),
        sensor_id="toy-sensor",
        viewing_geometry=viewing,
        band_meta=band_meta,
        quality_masks={"valid": np.array([True, False, True])},
        ancillary={"note": "roundtrip"},
    )

    chip = sample.to_chip()
    restored = Sample.from_chip(
        chip,
        wavelengths,
        sensor_id=sample.sensor_id,
        kind=sample.spectrum.kind,
        viewing_geometry=asdict(viewing),
        band_meta=band_meta,
        quality_masks=sample.quality_masks,
        ancillary=sample.ancillary,
    )

    assert restored.sensor_id == sample.sensor_id
    np.testing.assert_array_equal(restored.spectrum.values, sample.spectrum.values)
    assert restored.viewing_geometry == viewing
    assert restored.band_meta is not None
    np.testing.assert_array_equal(restored.band_meta.center_nm, sample.band_meta.center_nm)
    assert (
        restored.quality_masks["valid"][1] is False or restored.quality_masks["valid"][1] == False
    )
    assert restored.ancillary["note"] == "roundtrip"
