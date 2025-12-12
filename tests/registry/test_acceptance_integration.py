import numpy as np
import pytest

from alchemi.data import datasets as ds_module
from alchemi.registry.acceptance import AcceptanceVerdict
from alchemi.registry.sensors import DEFAULT_SENSOR_REGISTRY
from alchemi.spectral import BandMetadata, Sample, Spectrum
from alchemi.spectral.srf import SRFMatrix as DenseSRFMatrix
from alchemi.types import ValueUnits


class _FakeCatalog:
    def __init__(self, paths: list[str]):
        self.paths = paths

    def get_scenes(self, split: str, sensor_id: str, task: str) -> list[str]:  # noqa: ARG002
        return self.paths


def _bad_emit_sample(offset_nm: float = 40.0) -> Sample:
    spec = DEFAULT_SENSOR_REGISTRY.get_sensor("emit")
    centers = spec.band_centers_nm + offset_nm
    values = np.ones_like(centers)
    band_meta = BandMetadata(center_nm=centers, width_nm=spec.band_widths_nm, valid_mask=np.ones_like(centers, dtype=bool))
    srf_matrix = DenseSRFMatrix(wavelength_nm=centers, matrix=np.eye(centers.size, dtype=float))
    spectrum = Spectrum(
        wavelength_nm=centers,
        values=values,
        kind="radiance",
        units=ValueUnits.RADIANCE_W_M2_SR_NM,
    )
    return Sample(
        spectrum=spectrum,
        sensor_id="emit",
        band_meta=band_meta,
        srf_matrix=srf_matrix,
        quality_masks={"valid_band": np.ones_like(centers, dtype=bool)},
        ancillary={"source_path": "fake"},
    )


def test_catalog_dataset_blocks_rejected_sensor(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _FakeCatalog(["fake_scene.nc"])
    monkeypatch.setattr(ds_module, "iter_emit_pixels", lambda _path: [_bad_emit_sample()])

    with pytest.raises(RuntimeError):
        ds_module.EmitGasDataset(split="train", catalog=catalog)


def test_catalog_dataset_allows_force_override(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _FakeCatalog(["fake_scene.nc"])
    monkeypatch.setattr(ds_module, "iter_emit_pixels", lambda _path: [_bad_emit_sample()])

    dataset = ds_module.EmitGasDataset(split="train", catalog=catalog, force_accept_sensor=True)
    item = dataset[0]

    assert item["meta"]["sensor_acceptance"] == AcceptanceVerdict.REJECT.value
    assert item["meta"]["sensor_acceptance_rejections"]
