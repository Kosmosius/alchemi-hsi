import numpy as np
import pytest
import torch
import xarray as xr

from alchemi.data import adapters as adapter_module
from alchemi.data.adapters import emit as emit_adapter
from alchemi.data.adapters import hytes as hytes_adapter
from alchemi.data import datasets as ds_module
from alchemi.data.io.hytes import HYTES_BAND_COUNT, HYTES_WAVELENGTHS_NM
from alchemi.physics.planck import bt_sample_to_radiance_sample, radiance_sample_to_bt_sample
from alchemi.physics.rad_reflectance import (
    radiance_sample_to_toa_reflectance,
    toa_reflectance_sample_to_radiance,
)
from alchemi.spectral import BandMetadata, Sample, Spectrum, ViewingGeometry
from alchemi.types import QuantityKind, RadianceUnits, ReflectanceUnits, ValueUnits, WavelengthGrid


@pytest.fixture()
def flat_emit_scene(tmp_path):
    wavelengths = np.array([500.0, 600.0], dtype=np.float64)
    radiance_value = 10.0
    radiance = np.full((1, 1, wavelengths.size), radiance_value, dtype=np.float64)
    band_mask = np.array([True, True], dtype=bool)
    ds = xr.Dataset(
        {
            "radiance": (("y", "x", "band"), radiance),
            "band_mask": (("band",), band_mask),
        },
        coords={"wavelength_nm": ("band", wavelengths)},
        attrs={
            "solar_zenith": 0.0,
            "earth_sun_distance_au": 1.0,
            "radiance_units": ValueUnits.RADIANCE_W_M2_SR_NM.value,
        },
    )
    path = tmp_path / "emit.nc"
    ds.to_netcdf(path)
    return path, wavelengths, radiance_value


@pytest.fixture()
def hytes_scene(tmp_path):
    bt = np.full((1, 1, HYTES_BAND_COUNT), 295.0, dtype=np.float64)
    ds = xr.Dataset(
        {"brightness_temp": (("y", "x", "band"), bt)},
        coords={"wavelength_nm": ("band", HYTES_WAVELENGTHS_NM)},
    )
    path = tmp_path / "hytes.nc"
    ds.to_netcdf(path)
    return path


def test_radiance_reflectance_roundtrip_on_adapter_sample(monkeypatch, flat_emit_scene):
    path, wavelengths, radiance_value = flat_emit_scene

    monkeypatch.setattr(emit_adapter, "load_emit_l1b", lambda _path: xr.open_dataset(_path))
    monkeypatch.setattr(emit_adapter.srfs, "get_sensor_srf", lambda *_args, **_kwargs: None)

    samples = list(emit_adapter.iter_emit_pixels(str(path), srf_blind=True))
    assert len(samples) == 1
    radiance_sample = samples[0]

    esun_ref = Spectrum(
        wavelengths=WavelengthGrid(wavelengths),
        values=np.full_like(wavelengths, np.pi * radiance_value),
        kind=QuantityKind.RADIANCE,
        units=RadianceUnits.W_M2_SR_NM,
        meta={"quantity": "irradiance"},
    )

    reflect_sample = radiance_sample_to_toa_reflectance(radiance_sample, esun_ref=esun_ref)

    assert reflect_sample.spectrum.kind == QuantityKind.REFLECTANCE
    assert reflect_sample.spectrum.units == ReflectanceUnits.FRACTION
    assert radiance_sample.band_meta is reflect_sample.band_meta
    np.testing.assert_array_equal(reflect_sample.band_meta.center_nm, wavelengths)
    np.testing.assert_array_equal(reflect_sample.band_meta.valid_mask, radiance_sample.band_meta.valid_mask)
    for name, mask in radiance_sample.quality_masks.items():
        np.testing.assert_array_equal(reflect_sample.quality_masks[name], mask)

    recovered = toa_reflectance_sample_to_radiance(reflect_sample, esun_ref=esun_ref)
    np.testing.assert_allclose(recovered.spectrum.values, radiance_sample.spectrum.values)


def test_hytes_bt_radiance_roundtrip_preserves_metadata(monkeypatch, hytes_scene):
    path = hytes_scene

    monkeypatch.setattr(hytes_adapter, "load_hytes_l1b_bt", lambda _path: xr.open_dataset(_path))
    monkeypatch.setattr(hytes_adapter.srfs, "get_srf", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hytes_adapter, "default_band_widths", lambda _sensor, wl: np.ones_like(wl))

    bt_sample = next(iter(hytes_adapter.iter_hytes_pixels(str(path), srf_blind=True)))
    radiance_sample = bt_sample_to_radiance_sample(bt_sample)
    recovered = radiance_sample_to_bt_sample(radiance_sample)

    np.testing.assert_allclose(recovered.spectrum.values, bt_sample.spectrum.values, rtol=1e-6, atol=1e-6)
    assert recovered.sensor_id == bt_sample.sensor_id
    assert recovered.geo == bt_sample.geo
    assert recovered.viewing_geometry == bt_sample.viewing_geometry
    assert recovered.band_meta is bt_sample.band_meta
    np.testing.assert_array_equal(recovered.band_meta.center_nm, bt_sample.band_meta.center_nm)
    np.testing.assert_array_equal(recovered.band_meta.valid_mask, bt_sample.band_meta.valid_mask)
    for name, mask in bt_sample.quality_masks.items():
        np.testing.assert_array_equal(recovered.quality_masks[name], mask)
    assert recovered.ancillary == bt_sample.ancillary


def test_dataset_packing_consistency(monkeypatch, tmp_path):
    wavelengths = np.array([500.0, 600.0, 700.0], dtype=np.float64)
    radiance = np.stack(
        [np.array([1.0, 2.0, 3.0], dtype=np.float64), np.array([4.0, 5.0, 6.0], dtype=np.float64)],
        axis=1,
    ).reshape(1, 2, -1)
    mask = np.array([True, False, True], dtype=bool)

    ds = xr.Dataset(
        {
            "radiance": (("y", "x", "band"), radiance),
            "band_mask": (("band",), mask),
        },
        coords={"wavelength_nm": ("band", wavelengths)},
    )
    path = tmp_path / "emit.nc"
    ds.to_netcdf(path)

    monkeypatch.setattr(emit_adapter, "load_emit_l1b", lambda _path: xr.open_dataset(_path))
    monkeypatch.setattr(emit_adapter.srfs, "get_sensor_srf", lambda *_args, **_kwargs: None)

    samples = list(emit_adapter.iter_emit_pixels(str(path), srf_blind=True))

    spectrum_ds = ds_module.SpectrumDataset(samples)
    item0 = spectrum_ds[0]
    assert item0["wavelengths"].shape == (wavelengths.size,)
    assert item0["values"].shape == (wavelengths.size,)
    assert item0["mask"].shape == (wavelengths.size,)
    np.testing.assert_array_equal(item0["mask"].numpy(), samples[0].band_meta.valid_mask)
    assert item0["kind"] == samples[0].spectrum.kind
    assert item0["meta"]["sensor_id"] == samples[0].sensor_id

    paired = ds_module.PairingDataset(samples, samples)
    pair_item = paired[0]
    assert pair_item["field"]["wavelengths"].shape == (wavelengths.size,)
    assert pair_item["lab"]["mask"].shape == (wavelengths.size,)
    np.testing.assert_array_equal(pair_item["field"]["mask"].numpy(), samples[0].band_meta.valid_mask)

    loader_ds = xr.Dataset(
        {
            "radiance": (("y", "x", "band"), radiance),
            "band_mask": (("band",), mask),
        },
        coords={"wavelength_nm": ("band", wavelengths)},
    )
    ds_module.RealMAEDataset._LOADERS["flat"] = lambda _path: loader_ds
    realmae = ds_module.RealMAEDataset("flat", "ignored", patch_size=1, max_patches=1)
    mae_item = realmae[0]
    assert mae_item["tokens"].shape == (1, wavelengths.size)
    assert mae_item["wavelengths"].shape == (wavelengths.size,)
    assert mae_item["band_mask"].shape == (wavelengths.size,)

    class _Catalog:
        def __init__(self, paths):
            self.paths = paths

        def get_scenes(self, _split, _sensor_id, _task):
            return self.paths

    catalog = _Catalog([str(path)])

    monkeypatch.setattr(ds_module, "iter_enmap_pixels", lambda _p: [samples[0]])
    monkeypatch.setattr(ds_module, "iter_emit_pixels", lambda _p: [samples[0]])
    monkeypatch.setattr(ds_module, "iter_aviris_ng_pixels", lambda _p: [samples[0]])
    monkeypatch.setattr(ds_module, "iter_hytes_pixels", lambda _p: [samples[0]])

    se_dataset = ds_module.SpectralEarthDataset(split="train", catalog=catalog, patch_size=1, stride=1)
    se_item = se_dataset[0]
    assert se_item["chip"].shape == (1, 1, wavelengths.size)
    np.testing.assert_array_equal(se_item["wavelengths"].numpy(), samples[0].spectrum.wavelength_nm)

    solids_dataset = ds_module.EmitSolidsDataset(split="train", catalog=catalog, patch_size=1, stride=1)
    solids_item = solids_dataset[0]
    assert solids_item["chip"].shape == (1, 1, wavelengths.size)
    np.testing.assert_array_equal(solids_item["wavelengths"].numpy(), samples[0].spectrum.wavelength_nm)

    gas_dataset = ds_module.EmitGasDataset(split="train", catalog=catalog)
    gas_item = gas_dataset[0]
    assert gas_item["values"].shape == (wavelengths.size,)
    np.testing.assert_array_equal(gas_item["quality_masks"]["valid_band"].numpy(), samples[0].band_meta.valid_mask)

    aviris_gas_dataset = ds_module.AvirisGasDataset(split="train", catalog=catalog)
    aviris_item = aviris_gas_dataset[0]
    assert aviris_item["values"].shape == (wavelengths.size,)
    np.testing.assert_array_equal(aviris_item["quality_masks"]["valid_band"].numpy(), samples[0].band_meta.valid_mask)

    hytes_dataset = ds_module.HytesDataset(split="train", catalog=catalog, patch_size=1, stride=1)
    hytes_item = hytes_dataset[0]
    assert hytes_item["chip"].shape == (1, 1, wavelengths.size)
    np.testing.assert_array_equal(hytes_item["wavelengths"].numpy(), samples[0].spectrum.wavelength_nm)


def test_sample_chip_roundtrip_preserves_metadata():
    wavelengths = np.array([400.0, 500.0, 600.0], dtype=np.float64)
    values = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    valid_mask = np.array([True, False, True], dtype=bool)
    band_meta = BandMetadata(center_nm=wavelengths, width_nm=np.full_like(wavelengths, 10.0), valid_mask=valid_mask)
    quality_masks = {"valid_band": valid_mask, "custom": np.array([True, True, False])}
    ancillary = {"note": "test", "source_path": "fake"}
    viewing_geometry = ViewingGeometry(
        solar_zenith_deg=10.0,
        solar_azimuth_deg=20.0,
        view_zenith_deg=30.0,
        view_azimuth_deg=40.0,
        earth_sun_distance_au=1.0,
    )

    sample = Sample(
        spectrum=Spectrum(
            wavelength_nm=wavelengths,
            values=values,
            kind="radiance",
            units=ValueUnits.RADIANCE_W_M2_SR_NM,
        ),
        sensor_id="test_sensor",
        band_meta=band_meta,
        quality_masks=quality_masks,
        viewing_geometry=viewing_geometry,
        ancillary=ancillary,
    )

    chip = sample.to_chip()
    restored = Sample.from_chip(
        chip,
        wavelengths,
        sensor_id=sample.sensor_id,
        kind=sample.spectrum.kind,
        viewing_geometry=viewing_geometry,
        band_meta=band_meta,
        quality_masks=quality_masks,
        ancillary=ancillary,
    )

    np.testing.assert_array_equal(restored.spectrum.values, sample.spectrum.values)
    np.testing.assert_array_equal(restored.band_meta.center_nm, sample.band_meta.center_nm)
    np.testing.assert_array_equal(restored.band_meta.valid_mask, sample.band_meta.valid_mask)
    for name, mask in quality_masks.items():
        np.testing.assert_array_equal(restored.quality_masks[name], mask)
    assert restored.ancillary == sample.ancillary
    assert restored.viewing_geometry == sample.viewing_geometry
