from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import numpy as np
import torch
from torch.utils.data import Dataset

from alchemi.spectral import Sample, Spectrum
from alchemi.spectral.srf import SRFMatrix as DenseSRFMatrix
from alchemi.srf.synthetic import SyntheticSensorConfig
from alchemi.types import QuantityKind, SRFMatrix as LegacySRFMatrix
from alchemi.registry.acceptance import AcceptanceReport, AcceptanceVerdict, evaluate_sensor_acceptance
from alchemi.registry.sensors import DEFAULT_SENSOR_REGISTRY

from .adapters import iter_aviris_ng_pixels, iter_emit_pixels, iter_enmap_pixels, iter_hytes_pixels
from .catalog import SceneCatalog
from .io import load_emit_l1b
from .tiling import iter_tiles
from .transforms import SyntheticSensorProject


class SpectrumDataset(Dataset[dict[str, Any]]):
    """Simple Dataset wrapper for lists of spectral samples."""

    def __init__(self, samples: Sequence[Sample]):
        self.samples = list(samples)

    def __len__(self) -> int:  # pragma: no cover - trivial container
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, Any]:
        s = self.samples[i]
        w = torch.from_numpy(s.spectrum.wavelength_nm.astype("float32"))
        v = torch.from_numpy(s.spectrum.values.astype("float32"))
        band_mask = None
        if s.band_meta is not None:
            band_mask = s.band_meta.valid_mask
        if band_mask is None:
            band_mask = np.ones_like(s.spectrum.values, dtype=bool)
        m = torch.from_numpy(band_mask.astype("bool"))
        return {
            "wavelengths": w,
            "values": v,
            "mask": m,
            "kind": s.spectrum.kind,
            "meta": {"sensor_id": s.sensor_id, **s.ancillary},
        }


class PairingDataset(Dataset[dict[str, dict[str, Any]]]):
    """Dataset yielding dictionaries containing paired field/lab samples."""

    def __init__(self, field: Sequence[Sample], lab_conv: Sequence[Sample]):
        if len(field) != len(lab_conv):
            msg = "field and lab_conv must share length"
            raise ValueError(msg)
        self.field = list(field)
        self.lab = list(lab_conv)

    def __len__(self) -> int:  # pragma: no cover - trivial container
        return len(self.field)

    def __getitem__(self, i: int) -> dict[str, dict[str, Any]]:
        def pack(s: Sample) -> dict[str, Any]:
            w = torch.from_numpy(s.spectrum.wavelength_nm.astype("float32"))
            v = torch.from_numpy(s.spectrum.values.astype("float32"))
            band_mask = None
            if s.band_meta is not None:
                band_mask = s.band_meta.valid_mask
            if band_mask is None:
                band_mask = np.ones_like(s.spectrum.values, dtype=bool)
            m = torch.from_numpy(band_mask.astype("bool"))
            return {
                "wavelengths": w,
                "values": v,
                "mask": m,
                "kind": s.spectrum.kind,
                "meta": {"sensor_id": s.sensor_id, **s.ancillary},
            }

        return {"field": pack(self.field[i]), "lab": pack(self.lab[i])}


class SyntheticSensorDataset(Dataset[dict[str, Any]]):
    """Dataset wrapper that emits synthetic-sensor projections for MAE/tasks."""

    def __init__(
        self,
        lab_spectra: Sequence[Sample | Spectrum],
        config: SyntheticSensorConfig,
        *,
        sensor_id: str = "synthetic",
        quantity_kind: QuantityKind = QuantityKind.REFLECTANCE,
    ) -> None:
        if not lab_spectra:
            msg = "lab_spectra must contain at least one high-resolution sample"
            raise ValueError(msg)
        self.lab_spectra = list(lab_spectra)
        self.project = SyntheticSensorProject(config, sensor_id=sensor_id, quantity_kind=quantity_kind)

    def __len__(self) -> int:  # pragma: no cover - container
        return len(self.lab_spectra)

    def __getitem__(self, i: int) -> dict[str, Any]:
        lab = self.lab_spectra[i]
        spectrum = lab.spectrum if isinstance(lab, Sample) else lab  # type: ignore[assignment]
        sample = self.project(spectrum)

        bandwidths = sample.band_meta.width_nm if sample.band_meta is not None else None
        widths = (
            torch.from_numpy(np.asarray(bandwidths, dtype=np.float32))
            if bandwidths is not None
            else torch.empty(0)
        )
        srf = sample.srf_matrix.matrix if sample.srf_matrix is not None else None
        srf_tensor = (
            torch.from_numpy(np.asarray(srf, dtype=np.float32)) if srf is not None else None
        )

        return {
            "tokens": torch.from_numpy(sample.spectrum.values.astype("float32")),
            "wavelengths": torch.from_numpy(sample.spectrum.wavelength_nm.astype("float32")),
            "bandwidths": widths,
            "band_mask": torch.from_numpy(sample.band_meta.valid_mask.astype("bool")),
            "srf": srf_tensor,
            "sample": sample,
        }


class RealMAEDataset(Dataset[dict[str, torch.Tensor]]):
    """Tiny dataset that samples patches from real cube fixtures for MAE runs."""

    _LOADERS: ClassVar[dict[str, Any]] = {
        "emit_fixture": load_emit_l1b,
    }

    def __init__(
        self,
        dataset_name: str,
        path: str,
        *,
        patch_size: int = 2,
        max_patches: int = 32,
    ) -> None:
        if dataset_name not in self._LOADERS:
            msg = f"Unsupported dataset_name {dataset_name!r}"
            raise ValueError(msg)

        loader = self._LOADERS[dataset_name]
        ds = loader(path)

        if "radiance" not in ds:
            raise KeyError("Expected 'radiance' variable in dataset")

        data = ds["radiance"].values
        if data.ndim != 3:
            msg = "radiance data must be three-dimensional (y, x, band)"
            raise ValueError(msg)

        self.data = data.astype("float32", copy=False)
        self.wavelengths = torch.from_numpy(ds["wavelength_nm"].values.astype("float32"))
        band_mask = ds["band_mask"].values if "band_mask" in ds else None
        if band_mask is None:
            band_mask = torch.ones(self.data.shape[-1], dtype=torch.bool)
        else:
            band_mask = torch.from_numpy(band_mask.astype("bool"))
        self.band_mask = band_mask

        self.patch_size = max(1, patch_size)
        h, w, _ = self.data.shape
        effective_patch = min(self.patch_size, h, w)
        ys = range(0, h - effective_patch + 1, effective_patch)
        xs = range(0, w - effective_patch + 1, effective_patch)
        coords = [(y, x) for y in ys for x in xs]
        if not coords:
            coords = [(0, 0)]
        self.coords = coords[:max_patches]
        self.effective_patch = effective_patch

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        y, x = self.coords[index]
        patch = self.data[y : y + self.effective_patch, x : x + self.effective_patch]
        tokens = torch.from_numpy(patch.reshape(-1, patch.shape[-1]).astype("float32", copy=False))
        return {
            "tokens": tokens,
            "wavelengths": self.wavelengths,
            "band_mask": self.band_mask,
        }


# -----------------------------------------------------------------------------
# Catalog-backed datasets


def _pack_spectral_sample(
    sample: Sample,
    transform: Callable[[torch.Tensor], torch.Tensor] | None,
    acceptance_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    values = torch.from_numpy(sample.spectrum.values.astype("float32"))
    if transform is not None:
        values = transform(values)
    meta = {"sensor_id": sample.sensor_id, **sample.ancillary}
    if acceptance_meta:
        meta.update(acceptance_meta)
    return {
        "values": values,
        "wavelengths": torch.from_numpy(sample.spectrum.wavelength_nm.astype("float32")),
        "kind": sample.spectrum.kind,
        "quality_masks": {
            k: torch.from_numpy(v.astype("bool")) for k, v in sample.quality_masks.items()
        },
        "meta": meta,
    }


def _legacy_srf_from_sample(sample: Sample) -> LegacySRFMatrix | None:
    srf = sample.srf_matrix
    if srf is None:
        return None
    if isinstance(srf, LegacySRFMatrix):
        return srf
    if isinstance(srf, DenseSRFMatrix):
        centers = sample.band_meta.center_nm if sample.band_meta is not None else srf.wavelength_nm
        bands_nm = [np.asarray(srf.wavelength_nm, dtype=np.float64) for _ in range(srf.matrix.shape[0])]
        bands_resp = [np.asarray(row, dtype=np.float64) for row in srf.matrix]
        bad_mask = None
        if sample.band_meta is not None:
            bad_mask = ~np.asarray(sample.band_meta.valid_mask, dtype=bool)
        return LegacySRFMatrix(
            sensor=sample.sensor_id,
            centers_nm=np.asarray(centers, dtype=np.float64),
            bands_nm=bands_nm,
            bands_resp=bands_resp,
            bad_band_mask=bad_mask,
        )
    return None


class _BaseCatalogDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        split: str,
        sensor_id: str,
        task: str,
        catalog: SceneCatalog | None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None,
        force_accept_sensor: bool = False,
    ) -> None:
        self.catalog = catalog or SceneCatalog()
        self.scene_paths = self.catalog.get_scenes(split, sensor_id, task)
        self.transform = transform
        self.force_accept_sensor = force_accept_sensor
        self.samples: list[dict[str, Any]] = []
        self._acceptance_report: AcceptanceReport | None = None
        self._acceptance_logger = logging.getLogger(__name__)

    def __len__(self) -> int:  # pragma: no cover - container
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]

    def _maybe_check_acceptance(self, sample: Sample) -> None:
        if self._acceptance_report is not None:
            return

        legacy_srf = _legacy_srf_from_sample(sample)
        if legacy_srf is None:
            return

        try:
            spec = DEFAULT_SENSOR_REGISTRY.get_sensor(sample.sensor_id)
        except KeyError:
            return

        report = evaluate_sensor_acceptance(spec, legacy_srf)
        self._acceptance_report = report

        if report.verdict == AcceptanceVerdict.ACCEPT_WITH_WARNINGS:
            self._acceptance_logger.warning(
                "Sensor %s accepted with warnings: %s",
                sample.sensor_id,
                "; ".join(report.warnings),
            )
        if report.verdict == AcceptanceVerdict.REJECT:
            message = (
                f"Sensor {sample.sensor_id} failed acceptance: {', '.join(report.rejections)}."
                " Use force_accept_sensor to override."
            )
            if not self.force_accept_sensor:
                raise RuntimeError(message)
            self._acceptance_logger.warning(message)

    def _acceptance_meta(self) -> dict[str, Any]:
        if self._acceptance_report is None:
            return {}
        return {
            "sensor_acceptance": self._acceptance_report.verdict.value,
            "sensor_acceptance_warnings": list(self._acceptance_report.warnings),
            "sensor_acceptance_rejections": list(self._acceptance_report.rejections),
        }


class SpectralEarthDataset(_BaseCatalogDataset):
    """MAE pretraining dataset built from EnMAP-derived SpectralEarth patches."""

    def __init__(
        self,
        *,
        split: str,
        catalog: SceneCatalog | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        patch_size: int = 32,
        stride: int = 16,
        force_accept_sensor: bool = False,
    ) -> None:
        super().__init__(
            split=split,
            sensor_id="enmap",
            task="spectral_earth",
            catalog=catalog,
            transform=transform,
            force_accept_sensor=force_accept_sensor,
        )
        for scene in self.scene_paths:
            for sample in iter_enmap_pixels(str(scene)):
                self._maybe_check_acceptance(sample)
                # Build chips using a synthetic 2x2 window to keep stubs lightweight.
                chip = sample.spectrum.values.reshape(1, 1, -1)
                for tile in iter_tiles(chip, patch_size, stride):
                    tensor_chip = torch.from_numpy(tile.data.astype("float32"))
                    if transform is not None:
                        tensor_chip = transform(tensor_chip)
                    self.samples.append(
                        {
                            "chip": tensor_chip,
                            "wavelengths": torch.from_numpy(
                                sample.spectrum.wavelength_nm.astype("float32")
                            ),
                            "meta": self._acceptance_meta(),
                        }
                    )


class EmitSolidsDataset(_BaseCatalogDataset):
    """EMIT solids dataset with paired L2A/L2B labels."""

    def __init__(
        self,
        *,
        split: str,
        catalog: SceneCatalog | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        patch_size: int = 32,
        stride: int = 16,
        force_accept_sensor: bool = False,
    ) -> None:
        super().__init__(
            split=split,
            sensor_id="emit",
            task="emit_solids",
            catalog=catalog,
            transform=transform,
            force_accept_sensor=force_accept_sensor,
        )
        for scene in self.scene_paths:
            for sample in iter_emit_pixels(str(scene)):
                self._maybe_check_acceptance(sample)
                chip = sample.spectrum.values.reshape(1, 1, -1)
                for tile in iter_tiles(chip, patch_size, stride):
                    tensor_chip = torch.from_numpy(tile.data.astype("float32"))
                    if transform is not None:
                        tensor_chip = transform(tensor_chip)
                    self.samples.append(
                        {
                            "chip": tensor_chip,
                            "wavelengths": torch.from_numpy(
                                sample.spectrum.wavelength_nm.astype("float32")
                            ),
                            "labels": sample.ancillary.get("labels", {}),
                            "meta": self._acceptance_meta(),
                        }
                    )


class EmitGasDataset(_BaseCatalogDataset):
    """Gas detection dataset built from EMIT teacher outputs (CTMF/DOAS)."""

    def __init__(
        self,
        *,
        split: str,
        catalog: SceneCatalog | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        force_accept_sensor: bool = False,
    ) -> None:
        super().__init__(
            split=split,
            sensor_id="emit",
            task="emit_gas",
            catalog=catalog,
            transform=transform,
            force_accept_sensor=force_accept_sensor,
        )
        for scene in self.scene_paths:
            for sample in iter_emit_pixels(str(scene)):
                self._maybe_check_acceptance(sample)
                packed = _pack_spectral_sample(sample, transform, self._acceptance_meta())
                self.samples.append(packed)


class AvirisGasDataset(_BaseCatalogDataset):
    """Gas detection dataset for AVIRIS-NG scenes."""

    def __init__(
        self,
        *,
        split: str,
        catalog: SceneCatalog | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        force_accept_sensor: bool = False,
    ) -> None:
        super().__init__(
            split=split,
            sensor_id="aviris-ng",
            task="aviris_gas",
            catalog=catalog,
            transform=transform,
            force_accept_sensor=force_accept_sensor,
        )
        for scene in self.scene_paths:
            for sample in iter_aviris_ng_pixels(str(scene)):
                self._maybe_check_acceptance(sample)
                self.samples.append(_pack_spectral_sample(sample, transform, self._acceptance_meta()))


class HytesDataset(_BaseCatalogDataset):
    """LWIR brightness temperature chips."""

    def __init__(
        self,
        *,
        split: str,
        catalog: SceneCatalog | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        patch_size: int = 16,
        stride: int = 16,
        force_accept_sensor: bool = False,
    ) -> None:
        super().__init__(
            split=split,
            sensor_id="hytes",
            task="hytes_bt",
            catalog=catalog,
            transform=transform,
            force_accept_sensor=force_accept_sensor,
        )
        for scene in self.scene_paths:
            for sample in iter_hytes_pixels(str(scene)):
                self._maybe_check_acceptance(sample)
                chip = sample.spectrum.values.reshape(1, 1, -1)
                for tile in iter_tiles(chip, patch_size, stride):
                    tensor_chip = torch.from_numpy(tile.data.astype("float32"))
                    if transform is not None:
                        tensor_chip = transform(tensor_chip)
                    self.samples.append(
                        {
                            "chip": tensor_chip,
                            "wavelengths": torch.from_numpy(
                                sample.spectrum.wavelength_nm.astype("float32")
                            ),
                            "meta": {**sample.ancillary, **self._acceptance_meta()},
                        }
                    )
