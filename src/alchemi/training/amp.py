from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch


@contextlib.contextmanager
def autocast(enabled: bool = True, dtype: torch.dtype = torch.bfloat16) -> Iterator[None]:
    if enabled and torch.cuda.is_available():
        with torch.autocast("cuda", dtype=dtype):
            yield
    else:
        yield
