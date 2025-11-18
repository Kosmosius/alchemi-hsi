from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader, Dataset


def make_loader(
    dataset: Dataset[Any],
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
