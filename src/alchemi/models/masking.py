from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor


def _get_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


@dataclass
class MaskingConfig:
    """Configuration for spatial+spectral masking.

    Semantics:
      - spatial_mask_ratio: fraction of tokens to DROP (set False in mask).
      - spectral_mask_ratio: fraction of bands to DROP (set False in mask).
      - spectral_grouping: optional group size for band-wise masking.
      - mask_seed: RNG seed so we can ablate deterministically.
    """

    spatial_mask_ratio: float = 0.75
    spectral_mask_ratio: float = 0.5
    spectral_grouping: int | None = None
    mask_seed: int | None = None

    def persist(self, run_dir: Path) -> Path:
        """Persist JSON config under run_dir/mask_config.json."""
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = run_dir / "mask_config.json"
        cfg_path.write_text(json.dumps(asdict(self), indent=2))
        return cfg_path


def make_spatial_mask(num_tokens: int, mask_ratio_spatial: float, seed: int | None = None) -> Tensor:
    """Return a boolean mask of tokens to KEEP after spatial masking.

    True = keep token, False = masked-out token.
    """
    num_tokens = int(num_tokens)
    keep = torch.ones(num_tokens, dtype=torch.bool)
    num_mask = max(0, min(num_tokens, int(round(num_tokens * mask_ratio_spatial))))
    if num_mask == 0:
        return keep

    gen = _get_generator(seed)
    idx = torch.randperm(num_tokens, generator=gen)[:num_mask]
    keep[idx] = False
    return keep


def make_spectral_mask(
    B: int,
    mask_ratio_spectral: float,
    grouping: int | None = None,
    seed: int | None = None,
) -> Tensor:
    """Return a boolean mask of bands to KEEP after spectral masking.

    If ``grouping`` is provided, masking is applied independently to
    contiguous groups of that size to encourage spectral diversity.
    """
    B = int(B)
    keep = torch.ones(B, dtype=torch.bool)
    if B == 0 or mask_ratio_spectral <= 0.0:
        return keep

    gen = _get_generator(seed)
    group_size = grouping if grouping and grouping > 0 else B

    for start in range(0, B, group_size):
        end = min(B, start + group_size)
        n = end - start
        k = max(1, int(round(n * mask_ratio_spectral)))
        local_idx = torch.randperm(n, generator=gen)[:k]
        idx = local_idx + start
        keep[idx] = False

    return keep
