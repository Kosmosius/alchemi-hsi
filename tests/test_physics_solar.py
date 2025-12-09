import numpy as np
from datetime import datetime

from alchemi.physics.solar import (
    earth_sun_distance_au,
    earth_sun_distance_for_sample,
    esun_for_sample,
    get_reference_esun,
    project_esun_to_bands,
)
from alchemi.physics.resampling import generate_gaussian_srf
from alchemi.types import QuantityKind, RadianceUnits, Spectrum, WavelengthGrid, SRFMatrix


class _DummySample:
    def __init__(
        self,
        spectrum,
        band_meta=None,
        srf_matrix=None,
        ancillary=None,
        acquisition_time=None,
        viewing_geometry=None,
    ):
        self.spectrum = spectrum
        self.band_meta = band_meta
        self.srf_matrix = srf_matrix
        self.ancillary = ancillary or {}
        self.acquisition_time = acquisition_time
        self.viewing_geometry = viewing_geometry


class _DummyViewingGeometry:
    def __init__(self, distance):
        self.earth_sun_distance_au = distance


class _DummyBandMeta:
    def __init__(self, centers):
        self.center_nm = np.asarray(centers, dtype=float)


def test_reference_esun_properties():
    esun = get_reference_esun()
    wavelengths = esun.wavelengths.nm
    assert np.all(np.diff(wavelengths) > 0)
    assert esun.kind == QuantityKind.RADIANCE
    assert esun.meta.get("quantity") == "irradiance"
    assert np.all(esun.values > 0)
    assert esun.values.max() < 5.0


def test_project_esun_flat_response_remains_flat():
    wavelengths = np.linspace(400, 800, 401)
    flat_esun = Spectrum(
        wavelengths=WavelengthGrid(wavelengths),
        values=np.full_like(wavelengths, 2.5),
        kind=QuantityKind.RADIANCE,
        units=RadianceUnits.W_M2_SR_NM,
    )
    srf = generate_gaussian_srf("toy", (420, 780), num_bands=4, fwhm_nm=20)
    projected = project_esun_to_bands(flat_esun, srf)
    assert np.allclose(projected, 2.5, atol=1e-3)


def test_project_esun_narrow_band_matches_center_value():
    nm = np.linspace(590, 610, 81)
    values = 1.0 + 0.002 * (nm - 600.0)
    esun = Spectrum(
        wavelengths=WavelengthGrid(nm),
        values=values,
        kind=QuantityKind.RADIANCE,
        units=RadianceUnits.W_M2_SR_NM,
    )

    resp = np.exp(-0.5 * ((nm - 600.0) / 0.8) ** 2)
    narrow_srf = SRFMatrix(
        sensor="narrow",
        centers_nm=np.asarray([600.0]),
        bands_nm=[nm],
        bands_resp=[resp],
    ).normalize_rows_trapz()

    projected = project_esun_to_bands(esun, narrow_srf)
    assert np.allclose(projected, np.array([values[nm == 600.0][0]]), atol=1e-3)


def test_esun_for_sample_interpolates_without_srf():
    ref = get_reference_esun()
    band_centers = np.array([400.0, 550.0, 800.0])
    spectrum = Spectrum(
        wavelengths=WavelengthGrid(band_centers),
        values=np.ones_like(band_centers),
        kind=QuantityKind.RADIANCE,
        units=RadianceUnits.W_M2_SR_NM,
    )
    sample = _DummySample(spectrum, band_meta=_DummyBandMeta(band_centers))

    projected = esun_for_sample(sample, mode="interp")
    expected = np.interp(band_centers, ref.wavelengths.nm, ref.values)
    assert np.allclose(projected, expected)


def test_earth_sun_distance_ranges():
    assert 0.982 <= earth_sun_distance_au(doy=4) <= 0.985
    assert 1.015 <= earth_sun_distance_au(doy=186) <= 1.018
    assert 0.997 <= earth_sun_distance_au(doy=80) <= 1.003


def test_earth_sun_distance_for_sample_metadata_priority():
    vg = _DummyViewingGeometry(1.01)
    sample = _DummySample(None, viewing_geometry=vg)
    assert earth_sun_distance_for_sample(sample) == 1.01

    ancillary_sample = _DummySample(None, ancillary={"earth_sun_distance_au": 0.99})
    assert earth_sun_distance_for_sample(ancillary_sample) == 0.99

    dated_sample = _DummySample(None, acquisition_time=datetime(2024, 1, 3))
    assert earth_sun_distance_for_sample(dated_sample) == earth_sun_distance_au(
        dated_sample.acquisition_time
    )
