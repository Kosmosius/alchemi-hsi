import numpy as np
import pytest

from alchemi.spectral import BandMetadata, Sample, Spectrum, SRFMatrix, ViewingGeometry


def test_band_metadata_full_metadata_and_validation():
    center = np.array([400.0, 500.0, 600.0])
    width = np.array([10.0, 10.0, 10.0])
    valid_mask = np.array([True, False, True])
    srf_source = np.array(["catalog", "catalog", "manual"])

    meta = BandMetadata(center_nm=center, width_nm=width, valid_mask=valid_mask, srf_source=srf_source)

    np.testing.assert_array_equal(meta.center_nm, center)
    np.testing.assert_array_equal(meta.width_nm, width)
    np.testing.assert_array_equal(meta.valid_mask, valid_mask)
    np.testing.assert_array_equal(meta.srf_source, srf_source)

    meta.validate_length(3)


def test_band_metadata_optional_width_and_scalar_srf_source():
    center = np.array([700.0, 800.0])
    valid_mask = np.array([True, True])

    meta = BandMetadata(center_nm=center, width_nm=None, valid_mask=valid_mask, srf_source="official")

    assert meta.width_nm is None
    np.testing.assert_array_equal(meta.srf_source, np.array(["official", "official"], dtype=object))
    meta.validate_length(2)


def test_band_metadata_scalar_none_srf_source_defaults_to_empty():
    center = np.array([500.0, 600.0, 700.0])
    valid_mask = np.array([True, True, True])

    meta = BandMetadata(center_nm=center, width_nm=None, valid_mask=valid_mask, srf_source=None)

    np.testing.assert_array_equal(meta.srf_source, np.array(["", "", ""], dtype=object))
    meta.validate_length(3)


@pytest.mark.parametrize(
    "field,value,expected_message",
    [
        ("center_nm", BandMetadata(center_nm=[1.0], width_nm=[1.0], valid_mask=[True]), "center_nm"),
        (
            "valid_mask",
            BandMetadata(center_nm=[1.0, 2.0, 3.0], width_nm=[1.0, 1.0, 1.0], valid_mask=[True, False]),
            "valid_mask",
        ),
        (
            "srf_source",
            BandMetadata(
                center_nm=[1.0, 2.0, 3.0],
                width_nm=[1.0, 1.0, 1.0],
                valid_mask=[True, True, True],
                srf_source=["a", "a"],
            ),
            "srf_source",
        ),
    ],
)
def test_band_metadata_validate_length_errors(field, value, expected_message):
    with pytest.raises(ValueError) as excinfo:
        value.validate_length(3)
    assert expected_message in str(excinfo.value)


def test_sample_validate_success_with_matching_band_meta():
    wavelength = np.array([400.0, 500.0, 600.0])
    values = np.array([1.0, 2.0, 3.0])
    spectrum = Spectrum(wavelength_nm=wavelength, values=values, kind="radiance")
    band_meta = BandMetadata(center_nm=wavelength, width_nm=None, valid_mask=np.ones(3, dtype=bool))

    sample = Sample(spectrum=spectrum, sensor_id="sensor", band_meta=band_meta)

    sample.validate()  # Should not raise


def test_sample_validate_band_meta_length_mismatch():
    wavelength = np.array([400.0, 500.0, 600.0])
    spectrum = Spectrum(wavelength_nm=wavelength, values=np.ones(3), kind="radiance")
    band_meta = BandMetadata(center_nm=wavelength[:2], width_nm=None, valid_mask=np.ones(2, dtype=bool))

    with pytest.raises(ValueError):
        Sample(spectrum=spectrum, sensor_id="sensor", band_meta=band_meta)


def test_sample_validate_srf_matrix_wavelength_mismatch():
    wavelength = np.array([400.0, 500.0, 600.0])
    spectrum = Spectrum(wavelength_nm=wavelength, values=np.ones(3), kind="radiance")
    srf = SRFMatrix(wavelength_nm=np.array([400.0, 500.0]), matrix=np.ones((3, 2)))

    with pytest.raises(ValueError):
        Sample(spectrum=spectrum, sensor_id="sensor", srf_matrix=srf)


def test_sample_validate_srf_matrix_band_count_mismatch_with_band_meta():
    wavelength = np.array([400.0, 500.0, 600.0])
    spectrum = Spectrum(wavelength_nm=wavelength, values=np.ones(3), kind="radiance")
    band_meta = BandMetadata(center_nm=wavelength, width_nm=None, valid_mask=np.ones(3, dtype=bool))
    srf = SRFMatrix(wavelength_nm=wavelength, matrix=np.ones((2, 3)))

    with pytest.raises(ValueError):
        Sample(spectrum=spectrum, sensor_id="sensor", band_meta=band_meta, srf_matrix=srf)


def test_sample_validate_quality_masks_normalization_and_shape():
    wavelength = np.array([400.0, 500.0, 600.0])
    spectrum = Spectrum(wavelength_nm=wavelength, values=np.ones(3), kind="radiance")

    sample = Sample(
        spectrum=spectrum,
        sensor_id="sensor",
        quality_masks={"valid": [1, 0, 1]},
    )

    assert sample.quality_masks["valid"].dtype == bool
    np.testing.assert_array_equal(sample.quality_masks["valid"], np.array([True, False, True]))

    with pytest.raises(ValueError):
        Sample(
            spectrum=spectrum,
            sensor_id="sensor",
            quality_masks={"bad_length": np.array([True, False])},
        )

    with pytest.raises(ValueError):
        Sample(
            spectrum=spectrum,
            sensor_id="sensor",
            quality_masks={"not_1d": np.array([[True, False, True]])},
        )


def test_sample_validate_viewing_geometry_mapping_rejected_if_unconverted():
    wavelength = np.array([400.0, 500.0, 600.0])
    spectrum = Spectrum(wavelength_nm=wavelength, values=np.ones(3), kind="radiance")
    viewing = ViewingGeometry(10.0, 20.0, 30.0, 40.0, 1.0)

    sample = Sample(spectrum=spectrum, sensor_id="sensor", viewing_geometry=viewing)

    sample.viewing_geometry = {  # type: ignore[assignment]
        "solar_zenith_deg": 1.0,
        "solar_azimuth_deg": 2.0,
        "view_zenith_deg": 3.0,
        "view_azimuth_deg": 4.0,
        "earth_sun_distance_au": 1.0,
    }

    with pytest.raises(TypeError):
        sample.validate()
