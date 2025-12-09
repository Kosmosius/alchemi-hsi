from datetime import datetime

import numpy as np
import pytest

from alchemi.spectral import BandMetadata, GeoMeta, Sample, Spectrum, ViewingGeometry


def test_sample_validation_rejects_mismatched_quality_mask() -> None:
    spectrum = Spectrum(
        wavelength_nm=np.array([400.0, 500.0, 600.0]), values=np.ones(3), kind="radiance"
    )
    band_meta = BandMetadata(
        center_nm=spectrum.wavelength_nm,
        width_nm=np.array([10.0, 10.0, 10.0]),
        valid_mask=np.array([True, True, True]),
        srf_source=np.array(["catalog"] * 3),
    )

    with pytest.raises(ValueError):
        Sample(
            spectrum=spectrum,
            sensor_id="sensor",
            band_meta=band_meta,
            quality_masks={"bad": np.array([True, False])},
        )


def test_sample_serialization_roundtrip() -> None:
    spectrum = Spectrum(
        wavelength_nm=np.array([500.0, 600.0, 700.0]),
        values=np.array([0.1, 0.2, 0.3]),
        kind="reflectance",
    )
    viewing = ViewingGeometry(30.0, 120.0, 10.0, 200.0, 1.01)
    band_meta = BandMetadata(
        center_nm=spectrum.wavelength_nm,
        width_nm=np.array([5.0, 5.0, 5.0]),
        valid_mask=np.array([True, True, False]),
        srf_source=np.array(["catalog"] * 3),
    )
    quality_masks = {"band_mask": np.array([True, True, False])}
    ancillary = {"source_path": "scene.h5", "row": 1, "col": 2}

    sample = Sample(
        spectrum=spectrum,
        sensor_id="demo",
        acquisition_time=datetime(2024, 1, 1, 12, 0, 0),
        geo=GeoMeta(1.0, 2.0, 3.0),
        viewing_geometry=viewing,
        band_meta=band_meta,
        quality_masks=quality_masks,
        ancillary=ancillary,
    )

    restored = Sample.from_dict(sample.to_dict())

    assert restored.sensor_id == sample.sensor_id
    np.testing.assert_array_equal(restored.spectrum.values, sample.spectrum.values)
    np.testing.assert_array_equal(restored.band_meta.center_nm, band_meta.center_nm)
    assert restored.ancillary == ancillary
    assert restored.viewing_geometry == viewing
    assert restored.geo == sample.geo
