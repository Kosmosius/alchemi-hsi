from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import torch

from spectra.data.datamodule import SpectralEarthDataModule, pad_collate
from spectra.data.spectralearth import SpectralEarthDataset


def _write_sample(root: Path, idx: int, bands: int, sensor: str = "EnMAP") -> None:
    rng = np.random.default_rng(0)
    cube = rng.random((4, 4, bands), dtype="float32")
    wavelengths = np.linspace(400.0, 800.0, bands, dtype="float32")
    mask = np.ones(bands, dtype=bool)
    scene_id = f"scene_{idx}"
    root.mkdir(parents=True, exist_ok=True)
    np.savez(
        root / f"sample_{idx}.npz",
        cube=cube,
        wavelengths_nm=wavelengths,
        band_valid_mask=mask,
        sensor=sensor,
        scene_id=scene_id,
    )


def test_collate_padding() -> None:
    bands = [5, 7, 6]
    batch = []
    for i, b in enumerate(bands):
        cube = torch.ones((2, 2, b)) * (i + 1)
        wavelengths = torch.arange(b, dtype=torch.float32)
        mask = torch.ones(b, dtype=torch.bool)
        batch.append(
            {
                "cube": cube,
                "wavelengths_nm": wavelengths,
                "band_valid_mask": mask,
                "sensor": "EnMAP",
                "scene_id": f"scene_{i}",
            }
        )

    output = pad_collate(batch)
    assert output["cube"].shape == (3, 2, 2, max(bands))
    assert output["wavelengths_nm"].shape == (3, max(bands))
    assert output["band_valid_mask"].shape == (3, max(bands))
    assert torch.equal(output["band_valid_mask"], output["band_attention_mask"])
    assert torch.equal(output["band_valid_mask"], output["attention_mask"])
    for i, b in enumerate(bands):
        valid = output["band_valid_mask"][i]
        assert valid.sum().item() == b
        assert torch.all(valid[b:] == 0)



def spectraleart_loader_smoketest(tmp_path: Path) -> None:
    root = tmp_path / "spectralearth"
    root.mkdir()
    bands = [5, 6, 7, 8, 9]
    for idx, b in enumerate(bands):
        _write_sample(root, idx, b)

    dm = SpectralEarthDataModule(
        [root],
        batch_size=2,
        num_workers=0,
        split_seed=123,
        split_fracs=(0.6, 0.2, 0.2),
    )
    dm.setup()

    assert dm.train_set is not None
    assert dm.val_set is not None
    assert dm.test_set is not None
    assert len(dm.train_set) == 3
    assert len(dm.val_set) == 1
    assert len(dm.test_set) == 1

    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch["cube"].ndim == 4
    assert batch["cube"].shape[0] == dm.batch_size
    assert batch["wavelengths_nm"].shape[1] == batch["cube"].shape[-1]
    assert batch["band_valid_mask"].dtype == torch.bool
    assert torch.equal(batch["attention_mask"], batch["band_valid_mask"])
    for i in range(batch["cube"].shape[0]):
        valid = batch["band_valid_mask"][i]
        assert torch.all(valid[valid.sum().item() :] == 0)

    probes = dm.retrieval_probes
    if probes:
        anchor, candidates = probes[0]
        assert isinstance(anchor, int)
        assert candidates and all(isinstance(c, int) for c in candidates)
        assert anchor in range(len(dm.train_set))

    dm_again = SpectralEarthDataModule(
        [root],
        batch_size=2,
        num_workers=0,
        split_seed=123,
        split_fracs=(0.6, 0.2, 0.2),
    )
    dm_again.setup()
    assert dm.train_set is not None and dm_again.train_set is not None
    train_ids = [s.scene_id for s in dm.train_set.samples]
    train_ids_again = [s.scene_id for s in dm_again.train_set.samples]
    assert train_ids == train_ids_again

    images_seen = 0
    start = time.perf_counter()
    for i, batch in enumerate(loader):
        images_seen += batch["cube"].shape[0]
        if i >= 1:
            break
    elapsed = max(time.perf_counter() - start, 1e-6)
    images_per_s = images_seen / elapsed
    assert images_per_s > 0


def test_dataset_filters(tmp_path: Path) -> None:
    root = tmp_path / "spectralearth"
    root.mkdir()
    _write_sample(root, 0, 5, sensor="EMIT")
    _write_sample(root, 1, 6, sensor="AVIRIS")
    _write_sample(root, 2, 7, sensor="HyTES")
    _write_sample(root, 3, 8, sensor="Unknown")

    ds = SpectralEarthDataset([root], split="train", split_seed=0, split_fracs=(1, 0, 0))
    assert len(ds) == 3

    with pytest.raises(ValueError):
        SpectralEarthDataset(
            [root],
            split="train",
            split_seed=0,
            split_fracs=(1, 0, 0),
            allow_emit=False,
            allow_aviris=False,
            allow_hytes=False,
        )
