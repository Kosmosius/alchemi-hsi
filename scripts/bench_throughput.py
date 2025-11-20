from __future__ import annotations

import argparse

import csv
import time
from pathlib import Path
from typing import Sequence

import torch

from spectra.data.datamodule import SpectralEarthDataModule


@torch.no_grad()
def benchmark_throughput(
    roots: Sequence[str | Path],
    *,
    device: str,
    device_label: str | None,
    batch_size: int,
    num_workers: int,
    num_batches: int,
) -> dict[str, float | int | str]:
    """Measure SpectralEarth throughput in images/s, tokens/s, and GB/s."""

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.get_device_capability(torch_device)
        device_name = device_label or torch.cuda.get_device_name(torch_device)
    else:
        device_name = device_label or "CPU"

    dm = SpectralEarthDataModule(
        roots=roots,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    dm.setup()
    loader = dm.train_dataloader()

    total_images = 0
    total_tokens = 0
    total_bytes = 0

    start = time.perf_counter()
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break

        cube = batch["cube"].to(torch_device, non_blocking=True)
        band_mask = batch["band_valid_mask"].to(torch_device, non_blocking=True)

        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)

        bs, h, w, _ = cube.shape
        valid_bands = band_mask.sum(dim=1)  # [bs]
        tokens = (valid_bands * h * w).sum().item()

        total_images += bs
        total_tokens += int(tokens)
        total_bytes += cube.numel() * cube.element_size()

    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)

    elapsed = max(time.perf_counter() - start, 1e-6)

    return {
        "device": device_name,
        "device_arg": device,
        "batch_size": batch_size,
        "num_batches": batch_idx + 1,
        "images_per_s": total_images / elapsed,
        "tokens_per_s": total_tokens / elapsed,
        "gb_per_s": (total_bytes / (1024**3)) / elapsed,
    }


def _append_csv_row(path: Path, row: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SpectralEarth throughput")
    parser.add_argument("roots", nargs="+", help="Dataset root directories containing SpectralEarth NPZ samples")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. 'cuda', 'cuda:1', or 'cpu'",
    )
    parser.add_argument(
        "--device-label",
        default=None,
        help="Human-readable device label for CSV (e.g. 'A100', 'H100', 'RTX 4090')",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/benchmarks/spectralearth_throughput.csv"),
        help="CSV file to append benchmark results to",
    )
    args = parser.parse_args()

    results = benchmark_throughput(
        roots=args.roots,
        device=args.device,
        device_label=args.device_label,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_batches=args.num_batches,
    )

    print(
        f"Device: {results['device']} (arg={results['device_arg']})\n"
        f"Batch size: {results['batch_size']} | "
        f"Batches: {results['num_batches']}\n"
        f"Images/s: {results['images_per_s']:.2f}\n"
        f"Tokens/s: {results['tokens_per_s']:.2f}\n"
        f"GB/s: {results['gb_per_s']:.3f}"
    )

    _append_csv_row(args.output, results)
    print(f"Appended results to {args.output}")


if __name__ == "__main__":
    main()
