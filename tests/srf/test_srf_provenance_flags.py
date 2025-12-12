import numpy as np

from alchemi.registry import srfs
from alchemi.spectral.sample import BandMetadata
from alchemi.spectral.srf import SRFProvenance
from alchemi.srf.synthetic import make_virtual_sensor


def _band_meta_from_sensor(sensor_srf):
    return BandMetadata(
        center_nm=sensor_srf.band_centers_nm,
        width_nm=sensor_srf.band_widths_nm,
        valid_mask=np.ones(sensor_srf.band_count, dtype=bool),
        srf_source=sensor_srf.sensor_id,
        srf_provenance=sensor_srf.provenance.value,
        srf_approximate=sensor_srf.provenance != SRFProvenance.OFFICIAL,
    )


def test_emit_srf_provenance_official():
    sensor_srf = srfs.get_sensor_srf("emit")
    assert sensor_srf.provenance == SRFProvenance.OFFICIAL

    band_meta = _band_meta_from_sensor(sensor_srf)
    assert np.all(band_meta.srf_provenance == SRFProvenance.OFFICIAL.value)
    assert not np.any(band_meta.srf_approximate)


def test_hytes_gaussian_provenance_flagged_as_approximate():
    sensor_srf = srfs.get_sensor_srf("hytes")
    assert sensor_srf.provenance == SRFProvenance.GAUSSIAN

    band_meta = _band_meta_from_sensor(sensor_srf)
    assert np.all(band_meta.srf_provenance == SRFProvenance.GAUSSIAN.value)
    assert np.all(band_meta.srf_approximate)


def test_virtual_sensor_uses_synthetic_provenance():
    sensor_srf = make_virtual_sensor(
        wavelength_min_nm=400.0, wavelength_max_nm=900.0, band_count=12, grid_step_nm=1.0
    )
    assert sensor_srf.provenance == SRFProvenance.SYNTHETIC

    band_meta = _band_meta_from_sensor(sensor_srf)
    assert np.all(band_meta.srf_provenance == SRFProvenance.SYNTHETIC.value)
    assert np.all(band_meta.srf_approximate)
