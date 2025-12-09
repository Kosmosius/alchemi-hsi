import numpy as np

from alchemi.physics.rt_regime import (
    DEFAULT_AOD550_MAX,
    DEFAULT_PWV_MAX_CM,
    DEFAULT_SOLAR_ZENITH_MAX,
    DEFAULT_VIEW_ZENITH_MAX,
    SWIRRegime,
    attach_swir_regime,
    classify_swir_regime,
    swir_regime_for_sample,
    trusted_swir,
)
from alchemi.spectral.sample import Sample, ViewingGeometry
from alchemi.spectral.spectrum import Spectrum
from alchemi.types import WavelengthGrid


def _make_sample(**ancillary):
    spectrum = Spectrum.from_radiance(
        WavelengthGrid(np.array([1_000.0, 1_100.0])), np.array([0.1, 0.2])
    )
    viewing_geometry = ViewingGeometry(
        solar_zenith_deg=ancillary.pop("solar_zenith_deg", 0.0),
        solar_azimuth_deg=0.0,
        view_zenith_deg=ancillary.pop("view_zenith_deg", 0.0),
        view_azimuth_deg=0.0,
        earth_sun_distance_au=1.0,
    )
    return Sample(
        spectrum=spectrum,
        sensor_id="test",
        viewing_geometry=viewing_geometry,
        ancillary=ancillary,
    )


def test_classify_swir_regime_trusted_under_limits():
    regime = classify_swir_regime(
        solar_zenith_deg=40.0,
        view_zenith_deg=10.0,
        pwv_cm=2.0,
        aod_550=0.2,
        has_heavy_cloud_or_haze=False,
    )
    assert regime is SWIRRegime.TRUSTED


def test_classify_swir_regime_flags_heavy_conditions():
    assert (
        classify_swir_regime(
            solar_zenith_deg=DEFAULT_SOLAR_ZENITH_MAX + 1.0,
        )
        is SWIRRegime.HEAVY
    )
    assert (
        classify_swir_regime(
            view_zenith_deg=DEFAULT_VIEW_ZENITH_MAX + 5.0,
        )
        is SWIRRegime.HEAVY
    )
    assert (
        classify_swir_regime(
            pwv_cm=DEFAULT_PWV_MAX_CM + 0.5,
        )
        is SWIRRegime.HEAVY
    )
    assert (
        classify_swir_regime(
            aod_550=DEFAULT_AOD550_MAX + 0.1,
        )
        is SWIRRegime.HEAVY
    )
    assert classify_swir_regime(has_heavy_cloud_or_haze=True) is SWIRRegime.HEAVY


def test_single_heavy_condition_is_sufficient():
    regime = classify_swir_regime(
        solar_zenith_deg=30.0,
        view_zenith_deg=15.0,
        pwv_cm=DEFAULT_PWV_MAX_CM + 1.0,
        aod_550=0.2,
    )
    assert regime is SWIRRegime.HEAVY


def test_sample_regime_reads_geometry_and_ancillary():
    sample = _make_sample(pwv_cm=2.5, aod_550=0.2, has_heavy_cloud_or_haze=False)
    assert swir_regime_for_sample(sample) is SWIRRegime.TRUSTED

    heavy_sample = _make_sample(pwv_cm=DEFAULT_PWV_MAX_CM + 1.0, aod_550=0.2)
    assert swir_regime_for_sample(heavy_sample) is SWIRRegime.HEAVY


def test_sample_regime_gracefully_handles_missing_metadata():
    sample = _make_sample()
    sample.ancillary.clear()
    sample.viewing_geometry = None

    regime = swir_regime_for_sample(sample, solar_zenith_max=10.0)
    assert regime is SWIRRegime.TRUSTED


def test_attach_and_trusted_swir_helpers():
    sample = _make_sample(pwv_cm=10.0)
    tagged = attach_swir_regime(sample)

    assert tagged is not sample
    assert tagged.ancillary.get("swir_regime") == SWIRRegime.HEAVY.value
    assert trusted_swir(tagged) is False

    meta_only = {"aod_550": 0.1, "solar_zenith_deg": 20.0}
    assert trusted_swir(meta_only) is True
