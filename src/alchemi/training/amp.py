import contextlib

import torch


@contextlib.contextmanager
def autocast(enabled: bool = True, dtype=torch.bfloat16):
    if enabled and torch.cuda.is_available():
        with torch.autocast("cuda", dtype=dtype):
            yield
    else:
        yield
