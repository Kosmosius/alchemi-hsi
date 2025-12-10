import numpy as np

from alchemi.physics.solar import esun_for_sample, get_reference_esun, project_esun_to_bands
from alchemi.physics.swir import radiance_to_reflectance, reflectance_to_radiance
from alchemi.physics.resampling import generate_gaussian_srf, interpolate_to_centers
from alchemi.spectral.sample import BandMetadata
from alchemi.spectral.spectrum import Spectrum
from alchemi.types import QuantityKind, RadianceUnits, SRFMatrix, WavelengthGrid


class DummySample:
    def __init__(self, centers: np.ndarray, srf: SRFMatrix | None = None):
        self.band_meta = BandMetadata(center_nm=centers, width_nm=None, valid_mask=np.ones_like(centers, dtype=bool))
        self.srf_matrix = srf
        self.spectrum = Spectrum(
            wavelengths=WavelengthGrid(centers),
            values=np.ones_like(centers),
            kind=QuantityKind.RADIANCE,
            units=RadianceUnits.W_M2_SR_NM,
        )
        self.ancillary = {}
        self.viewing_geometry = None


def test_solar_reference_metadata_and_units():
    esun = get_reference_esun()
    assert esun.units == RadianceUnits.W_M2_SR_NM
    assert esun.meta.get("quantity") == "irradiance"
    assert esun.meta.get("units") == "W·m⁻²·nm⁻¹"
    assert np.all(np.diff(esun.wavelengths.nm) > 0)


def test_flat_spectrum_invariant_with_srf_resampled_esun():
    # constant Esun and reflectance should remain flat through radiance/reflectance conversion
    nm = np.linspace(400.0, 500.0, 101)
    esun_values = np.full_like(nm, 100.0)
    esun = Spectrum.from_radiance(WavelengthGrid(nm), esun_values, meta={"quantity": "irradiance"})

    srf = generate_gaussian_srf("toy", (400.0, 500.0), num_bands=6, fwhm_nm=5.0)
    esun_bands = project_esun_to_bands(esun, srf)

    reflectance = np.full(srf.centers_nm.shape, 0.2)
    radiance = reflectance_to_radiance(reflectance, esun_bands, cos_sun=1.0, tau=1.0, Lpath=0.0)
    recovered = radiance_to_reflectance(radiance, esun_bands, cos_sun=1.0, tau=1.0, Lpath=0.0)

    assert np.allclose(recovered, recovered[0], atol=1e-6)


def test_narrow_srf_matches_center_interpolation():
    nm = np.linspace(600.0, 700.0, 401)
    values = np.linspace(0.0, 1.0, nm.size)
    esun = Spectrum.from_radiance(WavelengthGrid(nm), values, meta={"quantity": "irradiance"})

    centers = np.array([620.0, 660.0, 690.0])
    bands_nm: list[np.ndarray] = []
    bands_resp: list[np.ndarray] = []
    sigma = 0.05
    for center in centers:
        nm_band = np.linspace(center - 0.3, center + 0.3, 61)
        resp = np.exp(-0.5 * ((nm_band - center) / sigma) ** 2)
        resp /= np.trapezoid(resp, x=nm_band)
        bands_nm.append(nm_band)
        bands_resp.append(resp)
    srf = SRFMatrix("narrow", centers, bands_nm, bands_resp)

    convolved = project_esun_to_bands(esun, srf)
    interpolated = interpolate_to_centers(esun, centers)

    assert np.allclose(convolved, interpolated.values, atol=5e-4)


def test_esun_for_sample_interpolation_when_srf_missing():
    centers = np.array([450.0, 500.0, 550.0])
    sample = DummySample(centers, srf=None)

    esun = esun_for_sample(sample, mode="interp")
    interpolated = interpolate_to_centers(get_reference_esun(), centers)

    assert np.allclose(esun, interpolated.values)

