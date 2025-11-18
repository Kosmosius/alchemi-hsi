import torch.distributed as dist


def init_ddp() -> None:
    if not (dist.is_available() and not dist.is_initialized()):
        return
    dist.init_process_group(backend="nccl")


def is_rank0() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
