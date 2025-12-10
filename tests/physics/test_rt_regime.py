import numpy as np

from alchemi.physics.rt_regime import (
    SWIRRegime,
    attach_swir_regime,
    classify_swir_regime,
    is_trusted_swir,
    swir_regime_for_sample,
)
from alchemi.spectral.sample import Sample, ViewingGeometry
from alchemi.spectral.spectrum import Spectrum
from alchemi.types import QuantityKind, RadianceUnits, WavelengthGrid


def _dummy_sample(**ancillary):
    spectrum = Spectrum(
        wavelengths=WavelengthGrid(np.array([400.0, 500.0])),
        values=np.ones(2),
        kind=QuantityKind.RADIANCE,
        units=RadianceUnits.W_M2_SR_NM,
    )
    return Sample(
        spectrum=spectrum,
        sensor_id="dummy",
        band_meta=None,
        srf_matrix=None,
        viewing_geometry=None,
        ancillary=ancillary,
    )


def test_classify_swir_regime_boundaries():
    assert (
        classify_swir_regime(
            solar_zenith_deg=59.0, view_zenith_deg=29.0, pwv_cm=3.9, aod_550=0.34
        )
        == SWIRRegime.TRUSTED
    )
    assert (
        classify_swir_regime(
            solar_zenith_deg=61.0, view_zenith_deg=20.0, pwv_cm=1.0, aod_550=0.2
        )
        == SWIRRegime.HEAVY
    )
    assert (
        classify_swir_regime(
            solar_zenith_deg=10.0, view_zenith_deg=31.0, pwv_cm=1.0, aod_550=0.2
        )
        == SWIRRegime.HEAVY
    )
    assert (
        classify_swir_regime(
            solar_zenith_deg=10.0, view_zenith_deg=20.0, pwv_cm=4.1, aod_550=0.2
        )
        == SWIRRegime.HEAVY
    )
    assert (
        classify_swir_regime(
            solar_zenith_deg=10.0, view_zenith_deg=20.0, pwv_cm=1.0, aod_550=0.36
        )
        == SWIRRegime.HEAVY
    )


def test_classify_vectorised_inputs():
    solar = np.array([30.0, 65.0])
    view = np.array([10.0, 40.0])
    pwv = np.array([2.0, 5.0])
    regimes = classify_swir_regime(solar_zenith_deg=solar, view_zenith_deg=view, pwv_cm=pwv)

    assert isinstance(regimes, np.ndarray)
    assert regimes.shape == solar.shape
    assert regimes[0] == SWIRRegime.TRUSTED
    assert regimes[1] == SWIRRegime.HEAVY


def test_sample_level_propagation_and_trust_helper():
    sample = _dummy_sample(solar_zenith_deg=50.0, view_zenith_deg=20.0, pwv_cm=2.0, aod_550=0.2)
    tagged = attach_swir_regime(sample)

    assert tagged.ancillary.get("swir_regime") == SWIRRegime.TRUSTED.value
    assert is_trusted_swir(tagged)

    heavy_sample = _dummy_sample(solar_zenith_deg=10.0, view_zenith_deg=20.0, pwv_cm=5.0)
    tagged_heavy = attach_swir_regime(heavy_sample)
    assert tagged_heavy.ancillary.get("swir_regime") == SWIRRegime.HEAVY.value
    assert not swir_regime_for_sample(tagged_heavy) == SWIRRegime.TRUSTED


def test_viewing_geometry_overrides_ancillary():
    ancillary = {"solar_zenith_deg": 20.0, "view_zenith_deg": 20.0, "pwv_cm": 1.0}
    sample = _dummy_sample(**ancillary)
    sample.viewing_geometry = ViewingGeometry(
        solar_zenith_deg=65.0,
        solar_azimuth_deg=0.0,
        view_zenith_deg=0.0,
        view_azimuth_deg=0.0,
        earth_sun_distance_au=1.0,
    )

    assert swir_regime_for_sample(sample) == SWIRRegime.HEAVY

