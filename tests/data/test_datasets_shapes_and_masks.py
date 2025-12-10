import numpy as np
import pytest
import torch
import xarray as xr

from alchemi.data import datasets as ds_module
from alchemi.spectral import BandMetadata, Sample, Spectrum
from alchemi.types import ValueUnits


pytestmark = pytest.mark.physics_and_metadata


def _make_sample(values: np.ndarray, wavelengths: np.ndarray, *, mask: np.ndarray | None = None) -> Sample:
    band_meta = None
    if mask is not None:
        band_meta = BandMetadata(center_nm=wavelengths, width_nm=None, valid_mask=mask)
    return Sample(
        spectrum=Spectrum(
            wavelength_nm=wavelengths, values=values, kind="radiance", units=ValueUnits.RADIANCE_W_M2_SR_NM
        ),
        sensor_id="test_sensor",
        band_meta=band_meta,
        quality_masks={"valid_band": mask} if mask is not None else {},
        ancillary={"source_path": "fake"},
    )


def test_spectrum_dataset_packs_shapes_and_masks():
    wavelengths = np.array([400.0, 500.0, 600.0], dtype=np.float64)
    mask = np.array([True, False, True])
    sample_with_mask = _make_sample(np.array([1.0, 2.0, 3.0]), wavelengths, mask=mask)
    sample_without_mask = _make_sample(np.array([0.1, 0.2, 0.3]), wavelengths)

    dataset = ds_module.SpectrumDataset([sample_with_mask, sample_without_mask])

    item0 = dataset[0]
    assert set(item0) == {"wavelengths", "values", "mask", "kind", "meta"}
    assert item0["wavelengths"].shape == (3,)
    assert item0["values"].shape == (3,)
    assert item0["mask"].shape == (3,)
    np.testing.assert_array_equal(item0["mask"].numpy(), mask)

    item1 = dataset[1]
    assert torch.all(item1["mask"])


def test_pairing_dataset_enforces_lengths_and_packs_samples():
    wavelengths = np.array([400.0, 500.0, 600.0], dtype=np.float64)
    field_sample = _make_sample(np.array([1.0, 2.0, 3.0]), wavelengths, mask=np.array([True, False, True]))
    lab_sample = _make_sample(np.array([0.5, 0.6, 0.7]), wavelengths)

    dataset = ds_module.PairingDataset([field_sample], [lab_sample])
    pair = dataset[0]
    assert set(pair) == {"field", "lab"}
    assert pair["field"]["values"].shape == (3,)
    assert pair["lab"]["values"].shape == (3,)
    np.testing.assert_array_equal(pair["field"]["mask"].numpy(), np.array([True, False, True]))
    assert torch.all(pair["lab"]["mask"])

    with pytest.raises(ValueError):
        ds_module.PairingDataset([field_sample], [])


def test_realmae_dataset_shapes(monkeypatch):
    wavelengths = np.array([400.0, 500.0, 600.0], dtype=np.float64)
    radiance = np.arange(2 * 2 * wavelengths.size, dtype=np.float64).reshape(2, 2, wavelengths.size)
    band_mask = np.array([True, False, True])

    ds = xr.Dataset(
        {
            "radiance": (("y", "x", "band"), radiance),
            "band_mask": (("band",), band_mask),
        },
        coords={"wavelength_nm": ("band", wavelengths)},
    )

    monkeypatch.setitem(ds_module.RealMAEDataset._LOADERS, "emit_fixture", lambda _path: ds)

    dataset = ds_module.RealMAEDataset("emit_fixture", "ignored", patch_size=2, max_patches=4)
    item = dataset[0]
    assert item["tokens"].shape == (4, wavelengths.size)
    assert item["wavelengths"].shape == (wavelengths.size,)
    assert item["band_mask"].shape == (wavelengths.size,)


class _FakeCatalog:
    def __init__(self, paths: list[str]):
        self.paths = paths

    def get_scenes(self, split: str, sensor_id: str, task: str) -> list[str]:
        return self.paths


def test_catalog_backed_datasets_shapes(monkeypatch):
    wavelengths = np.array([400.0, 500.0, 600.0], dtype=np.float64)
    mask = np.array([True, True, False])
    values = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    enmap_sample = _make_sample(values, wavelengths, mask=mask)
    emit_sample = _make_sample(values * 2, wavelengths, mask=mask)
    aviris_sample = _make_sample(values * 3, wavelengths, mask=mask)
    hytes_sample = _make_sample(values * 4, wavelengths, mask=mask)
    hytes_sample.spectrum.kind = "brightness_temperature"

    catalog = _FakeCatalog(["fake_scene.nc"])

    monkeypatch.setattr(ds_module, "iter_enmap_pixels", lambda _path: [enmap_sample])
    monkeypatch.setattr(ds_module, "iter_emit_pixels", lambda _path: [emit_sample])
    monkeypatch.setattr(ds_module, "iter_aviris_ng_pixels", lambda _path: [aviris_sample])
    monkeypatch.setattr(ds_module, "iter_hytes_pixels", lambda _path: [hytes_sample])

    se_dataset = ds_module.SpectralEarthDataset(
        split="train", catalog=catalog, patch_size=1, stride=1
    )
    se_item = se_dataset[0]
    assert se_item["chip"].shape == (1, 1, wavelengths.size)
    assert se_item["wavelengths"].shape == (wavelengths.size,)

    solids_dataset = ds_module.EmitSolidsDataset(
        split="train", catalog=catalog, patch_size=1, stride=1
    )
    solids_item = solids_dataset[0]
    assert solids_item["chip"].shape == (1, 1, wavelengths.size)
    assert solids_item["wavelengths"].shape == (wavelengths.size,)
    assert isinstance(solids_item["labels"], dict)

    gas_dataset = ds_module.EmitGasDataset(split="train", catalog=catalog)
    gas_item = gas_dataset[0]
    assert gas_item["values"].shape == (wavelengths.size,)
    assert set(gas_item["quality_masks"]) == {"valid_band"}

    aviris_gas_dataset = ds_module.AvirisGasDataset(split="train", catalog=catalog)
    aviris_gas_item = aviris_gas_dataset[0]
    assert aviris_gas_item["values"].shape == (wavelengths.size,)
    assert set(aviris_gas_item["quality_masks"]) == {"valid_band"}

    hytes_dataset = ds_module.HytesDataset(split="train", catalog=catalog, patch_size=1, stride=1)
    hytes_item = hytes_dataset[0]
    assert hytes_item["chip"].shape == (1, 1, wavelengths.size)
    assert hytes_item["wavelengths"].shape == (wavelengths.size,)
    assert "meta" in hytes_item
