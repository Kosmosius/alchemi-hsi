from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SpectralSample:
    path: Path
    sensor: str
    scene_id: str


class SpectralEarthDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Dataset for multi-sensor SpectralEarth/EnMAP-style spectral cubes.

    Each sample is stored as an ``.npz`` containing at minimum:

    - ``cube``: float32, shape [H, W, B]
    - ``wavelengths_nm``: float32, shape [B]

    Optional keys:

    - ``band_valid_mask``: bool, shape [B]
    - ``sensor``: string (scalar or 0-d array)
    - ``scene_id``: string (scalar or 0-d array)
    """

    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        split: str = "train",
        split_seed: int = 0,
        split_fracs: tuple[float, float, float] = (0.8, 0.1, 0.1),
        pattern: str = "*.npz",
        allow_emit: bool = True,
        allow_aviris: bool = True,
        allow_hytes: bool = True,
        probe_stride: int = 10,
        probe_candidates: int = 4,
    ) -> None:
        self.roots = [Path(r) for r in roots]
        self.split = split
        self.split_seed = split_seed
        self.split_fracs = split_fracs
        self.pattern = pattern
        self.allow_emit = allow_emit
        self.allow_aviris = allow_aviris
        self.allow_hytes = allow_hytes
        self.probe_stride = probe_stride
        self.probe_candidates = probe_candidates

        self.samples: list[SpectralSample] = self._load_samples()
        self.probe_pairs: list[tuple[int, list[int]]] = self._build_probe_pairs()

    # ------------------------------------------------------------------
    # Discovery & splits
    # ------------------------------------------------------------------
    def _load_samples(self) -> list[SpectralSample]:
        raw_paths: list[Path] = []
        for root in self.roots:
            raw_paths.extend(sorted(root.glob(self.pattern)))

        if not raw_paths:
            msg = f"No NPZ samples found under roots: {self.roots}"
            raise FileNotFoundError(msg)

        sensors_allowed: set[str] = set()
        if self.allow_emit:
            sensors_allowed.add("EMIT")
        if self.allow_aviris:
            sensors_allowed.add("AVIRIS")
            sensors_allowed.add("AVIRIS-NG")
        if self.allow_hytes:
            sensors_allowed.add("HyTES")
        sensors_allowed.add("EnMAP")

        dataset: list[SpectralSample] = []
        for p in raw_paths:
            with np.load(p, allow_pickle=True) as npz:
                sensor_raw = npz.get("sensor", "EnMAP")
                if isinstance(sensor_raw, np.ndarray):
                    sensor = str(sensor_raw.item())
                else:
                    sensor = str(sensor_raw)

                if sensor not in sensors_allowed:
                    continue

                scene_id_raw = npz.get("scene_id", p.stem)
                if isinstance(scene_id_raw, np.ndarray):
                    scene_id = str(scene_id_raw.item())
                else:
                    scene_id = str(scene_id_raw)

            dataset.append(SpectralSample(path=p, sensor=sensor, scene_id=scene_id))

        if not dataset:
            msg = "No samples matched allowed sensors."
            raise ValueError(msg)

        if not np.isclose(sum(self.split_fracs), 1.0):
            msg = "split_fracs must sum to 1.0"
            raise ValueError(msg)

        rng = np.random.default_rng(self.split_seed)
        indices = np.arange(len(dataset))
        rng.shuffle(indices)

        n_total = len(dataset)
        train_frac, val_frac, _test_frac = self.split_fracs
        n_train = int(train_frac * n_total)
        n_val = int(val_frac * n_total)
        n_test = n_total - n_train - n_val
        if n_test < 0:
            msg = "Invalid split_fracs; test split is negative."
            raise ValueError(msg)

        if self.split == "train":
            chosen = indices[:n_train]
        elif self.split == "val":
            chosen = indices[n_train : n_train + n_val]
        elif self.split == "test":
            chosen = indices[n_train + n_val : n_train + n_val + n_test]
        else:
            msg = f"Unknown split '{self.split}'"
            raise ValueError(msg)

        return [dataset[int(i)] for i in chosen]

    # ------------------------------------------------------------------
    # Retrieval probes
    # ------------------------------------------------------------------
    def _build_probe_pairs(self) -> list[tuple[int, list[int]]]:
        """Build simple anchor/candidate pairs for retrieval sanity checks."""

        pairs: list[tuple[int, list[int]]] = []
        if self.probe_stride <= 0 or self.probe_candidates <= 0:
            return pairs

        indices = list(range(len(self.samples)))
        for anchor in indices[:: self.probe_stride]:
            candidates = [i for i in indices if i != anchor][: self.probe_candidates]
            if candidates:
                pairs.append((anchor, candidates))
        return pairs

    def iter_probe_pairs(self) -> Iterable[tuple[int, list[int]]]:
        """Yield (anchor_idx, [candidate_idx...]) pairs for retrieval probes."""

        yield from self.probe_pairs

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def _load_npz(self, sample: SpectralSample) -> dict[str, torch.Tensor | str]:
        with np.load(sample.path, allow_pickle=True) as npz:
            cube = np.asarray(npz["cube"], dtype="float32")
            wavelengths = np.asarray(npz["wavelengths_nm"], dtype="float32")

            if cube.shape[-1] != wavelengths.shape[0]:
                msg = (
                    f"cube last dimension ({cube.shape[-1]}) must match wavelengths length "
                    f"({wavelengths.shape[0]})"
                )
                raise ValueError(msg)

            band_valid_mask = np.asarray(
                npz.get("band_valid_mask", np.ones_like(wavelengths, dtype=bool)),
                dtype=bool,
            )

        cube_t = torch.from_numpy(cube)
        wavelengths_t = torch.from_numpy(wavelengths)
        mask_t = torch.from_numpy(band_valid_mask.astype(bool))

        return {
            "cube": cube_t,
            "wavelengths_nm": wavelengths_t,
            "band_valid_mask": mask_t,
            "sensor": sample.sensor,
            "scene_id": sample.scene_id,
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        return self._load_npz(sample)
