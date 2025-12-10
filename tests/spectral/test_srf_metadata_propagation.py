import numpy as np
import xarray as xr

from alchemi.registry import srfs
from alchemi.srf.registry import sensor_srf_from_legacy
from alchemi.spectral.srf import SRFProvenance
from alchemi.data.adapters import emit
from alchemi.types import SRFMatrix as LegacySRFMatrix, ValueUnits


def _synthetic_legacy_srf(sensor: str, centers: np.ndarray, *, bad_mask: np.ndarray) -> LegacySRFMatrix:
    bands_nm = [np.asarray(centers, dtype=np.float64) for _ in centers]
    bands_resp = [
        np.exp(-0.5 * ((bands - center) ** 2) / 0.01)  # simple peaked response
        for bands, center in zip(bands_nm, centers, strict=True)
    ]
    return LegacySRFMatrix(
        sensor=sensor,
        centers_nm=np.asarray(centers, dtype=np.float64),
        bands_nm=bands_nm,
        bands_resp=bands_resp,
        bad_band_mask=np.asarray(bad_mask, dtype=bool),
        bad_band_windows_nm=[(float(centers[-1]) - 0.05, float(centers[-1]) + 0.05)],
    ).normalize_rows_trapz()


def test_sensor_srf_valid_mask_and_normalization():
    centers = np.array([1.0, 2.0])
    legacy = _synthetic_legacy_srf("unit-test", centers, bad_mask=np.array([False, True]))

    sensor_srf = sensor_srf_from_legacy(
        legacy, provenance=SRFProvenance.OFFICIAL, valid_mask=~legacy.bad_band_mask
    )

    assert sensor_srf.valid_mask.tolist() == [True, False]
    assert sensor_srf.meta.get("bad_band_windows_nm") == legacy.bad_band_windows_nm

    dense = sensor_srf.as_matrix()
    integrals = np.trapz(dense.matrix, x=dense.wavelength_nm, axis=1)
    assert np.allclose(integrals, 1.0)


def test_emit_adapter_propagates_srf_masks(monkeypatch):
    wavelengths = np.array([1.0, 1.5, 2.05])
    legacy = _synthetic_legacy_srf("emit", wavelengths, bad_mask=np.array([False, True, False]))
    sensor_srf = sensor_srf_from_legacy(
        legacy, provenance=SRFProvenance.OFFICIAL, valid_mask=~legacy.bad_band_mask
    )
    sensor_srf.meta["bad_band_windows_nm"] = [(2.0, 2.1)]

    monkeypatch.setattr(srfs, "get_sensor_srf", lambda *_, **__: sensor_srf)

    radiance = xr.DataArray(
        np.ones((1, 1, wavelengths.size), dtype=np.float64),
        dims=("y", "x", "band"),
        attrs={"units": ValueUnits.RADIANCE_W_M2_SR_NM.value},
    )
    ds = xr.Dataset({"radiance": radiance}).assign_coords(
        wavelength_nm=("band", np.asarray(wavelengths, dtype=np.float64))
    )

    monkeypatch.setattr(emit, "load_emit_l1b", lambda *_args, **_kwargs: ds)

    samples = list(emit.iter_emit_pixels("placeholder", include_quality=False, srf_blind=False))
    assert len(samples) == 1

    sample = samples[0]
    assert sample.band_meta.valid_mask.tolist() == [True, False, False]
    assert sample.quality_masks["srf_bad_band"].tolist() == [False, True, False]
    assert sample.quality_masks["srf_bad_window"].tolist() == [False, False, True]
