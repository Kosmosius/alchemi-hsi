import torch.distributed as dist


def init_ddp():
    if not (dist.is_available() and not dist.is_initialized()):
        return
    dist.init_process_group(backend="nccl")


def is_rank0():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
