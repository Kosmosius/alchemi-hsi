from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

try:  # pragma: no cover - optional dependency
    from lightning import LightningDataModule
except Exception:  # pragma: no cover - fallback
    try:
        from pytorch_lightning import LightningDataModule  # type: ignore[import]
    except Exception:  # pragma: no cover - last resort stub

        class LightningDataModule:  # type: ignore[misc]
            """Minimal stub to allow import without Lightning installed."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__()

            def setup(self, stage: str | None = None) -> None:  # noqa: D401
                """Placeholder setup."""

            def train_dataloader(self) -> Iterable[Any]:  # noqa: D401
                """Placeholder train loader."""

            def val_dataloader(self) -> Iterable[Any]:  # noqa: D401
                """Placeholder val loader."""

            def test_dataloader(self) -> Iterable[Any]:  # noqa: D401
                """Placeholder test loader."""


from .spectralearth import SpectralEarthDataset
from spectra.utils.seed import seed_everything


# ---------------------------------------------------------------------------
# Collate utilities for variable-band cubes
# ---------------------------------------------------------------------------
def pad_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable band dimension to the per-batch maximum and emit masks.

    Input samples are dicts with keys:

    - ``cube``: [H, W, B_i]
    - ``wavelengths_nm``: [B_i]
    - ``band_valid_mask``: [B_i]
    - ``sensor``: str
    - ``scene_id``: str
    """

    if not batch:
        msg = "Empty batch passed to pad_collate"
        raise ValueError(msg)

    max_bands = max(item["cube"].shape[-1] for item in batch)

    batch_cubes: list[torch.Tensor] = []
    batch_wavelengths: list[torch.Tensor] = []
    batch_masks: list[torch.Tensor] = []
    sensors: list[str] = []
    scene_ids: list[str] = []

    for item in batch:
        cube = item["cube"]
        wavelengths = item["wavelengths_nm"]
        mask = item["band_valid_mask"]

        pad_bands = max_bands - cube.shape[-1]
        if pad_bands < 0:
            msg = "pad_collate encountered sample with more bands than max_bands"
            raise ValueError(msg)

        if pad_bands > 0:
            cube = torch.nn.functional.pad(cube, (0, pad_bands))
            wavelengths = torch.nn.functional.pad(wavelengths, (0, pad_bands))
            mask = torch.nn.functional.pad(mask, (0, pad_bands))

        batch_cubes.append(cube)
        batch_wavelengths.append(wavelengths)
        batch_masks.append(mask.to(torch.bool))
        sensors.append(str(item["sensor"]))
        scene_ids.append(str(item["scene_id"]))

    cube_tensor = torch.stack(batch_cubes, dim=0)
    wavelengths_tensor = torch.stack(batch_wavelengths, dim=0)
    mask_tensor = torch.stack(batch_masks, dim=0)

    return {
        "cube": cube_tensor,
        "wavelengths_nm": wavelengths_tensor,
        "band_valid_mask": mask_tensor,
        # Aliases for downstream models / benchmarks:
        "band_attention_mask": mask_tensor,
        "attention_mask": mask_tensor,
        "sensor": sensors,
        "scene_id": scene_ids,
    }


# ---------------------------------------------------------------------------
# SpectralEarth data module (used by tests)
# ---------------------------------------------------------------------------
class SpectralEarthDataModule(LightningDataModule):
    """Lightning DataModule wrapping :class:`SpectralEarthDataset`."""

    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        batch_size: int = 4,
        num_workers: int = 0,
        prefetch_factor: int | None = 2,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        split_seed: int = 0,
        split_fracs: tuple[float, float, float] = (0.8, 0.1, 0.1),
        allow_emit: bool = True,
        allow_aviris: bool = True,
        allow_hytes: bool = True,
        probe_stride: int = 10,
        probe_candidates: int = 4,
    ) -> None:
        super().__init__()
        self.roots = [Path(r) for r in roots]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.split_seed = split_seed
        self.split_fracs = split_fracs
        self.allow_emit = allow_emit
        self.allow_aviris = allow_aviris
        self.allow_hytes = allow_hytes
        self.probe_stride = probe_stride
        self.probe_candidates = probe_candidates

        self.train_set: SpectralEarthDataset | None = None
        self.val_set: SpectralEarthDataset | None = None
        self.test_set: SpectralEarthDataset | None = None

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------
    def _make_dataset(self, split: str) -> SpectralEarthDataset:
        return SpectralEarthDataset(
            self.roots,
            split=split,
            split_seed=self.split_seed,
            split_fracs=self.split_fracs,
            allow_emit=self.allow_emit,
            allow_aviris=self.allow_aviris,
            allow_hytes=self.allow_hytes,
            probe_stride=self.probe_stride if split == "train" else 0,
            probe_candidates=self.probe_candidates,
        )

    def setup(self, stage: str | None = None) -> None:  # pragma: no cover - wiring
        if stage in (None, "fit", "train"):
            self.train_set = self._make_dataset("train")
            self.val_set = self._make_dataset("val")
        if stage in (None, "validate") and self.val_set is None:
            self.val_set = self._make_dataset("val")
        if stage in (None, "test"):
            self.test_set = self._make_dataset("test")

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    def _loader(self, dataset: SpectralEarthDataset | None, shuffle: bool) -> DataLoader[dict[str, Any]]:
        if dataset is None:
            msg = "DataModule.setup() must be called before requesting dataloaders"
            raise RuntimeError(msg)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=pad_collate,
        )

    def train_dataloader(self) -> DataLoader[dict[str, Any]]:
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader[dict[str, Any]]:
        return self._loader(self.val_set, shuffle=False)

    def test_dataloader(self) -> DataLoader[dict[str, Any]]:
        return self._loader(self.test_set, shuffle=False)

    # ------------------------------------------------------------------
    # Retrieval probes
    # ------------------------------------------------------------------
    @property
    def retrieval_probes(self) -> list[tuple[int, list[int]]]:
        """Expose retrieval probe pairs from the training split."""
        if self.train_set is None:
            return []
        return list(self.train_set.iter_probe_pairs())


# ---------------------------------------------------------------------------
# Synthetic variable-band data for MAE smoke tests
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    batch_size: int = 2
    num_workers: int = 0
    dataset_path: str = "/path/to/dataset"
    height: int = 8
    width: int = 8
    bands: tuple[int, ...] = (224, 285)


class RandomCubeDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    """Produces synthetic hyperspectral cubes with per-sample band counts."""

    def __init__(self, config: DataConfig) -> None:
        super().__init__()
        self.config = config
        self.step = 0

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        seed_everything(0)
        while True:
            bands = self.config.bands[self.step % len(self.config.bands)]
            self.step += 1
            cube = torch.randn(
                self.config.batch_size,
                self.config.height,
                self.config.width,
                bands,
            )
            wavelengths = torch.rand(self.config.batch_size, bands) * (2500 - 400) + 400
            wavelengths, _ = torch.sort(wavelengths, dim=-1)
            yield cube, wavelengths


class SyntheticInfiniteDataModule(LightningDataModule):
    """Endless synthetic data iterator used for throughput tests."""

    def __init__(self, config: DataConfig | None = None) -> None:
        super().__init__()
        self.config = config or DataConfig()
        self.dataset = RandomCubeDataset(self.config)

    def setup(self, stage: str | None = None) -> None:  # pragma: no cover - nothing to do
        seed_everything(0)

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        return DataLoader(
            self.dataset,
            batch_size=None,
            num_workers=self.config.num_workers,
        )


__all__ = [
    "pad_collate",
    "SpectralEarthDataModule",
    "DataConfig",
    "RandomCubeDataset",
    "SyntheticInfiniteDataModule",
]
