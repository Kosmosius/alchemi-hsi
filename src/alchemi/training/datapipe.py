import torch
from torch.utils.data import DataLoader


def make_loader(dataset, batch_size: int = 64, shuffle: bool = True, num_workers: int = 0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
