"""Propagate a cache failure to every tensor-parallel worker before returning."""


def any_rank_failed(failed, world_size):
    if type(world_size) is not int or world_size < 1:
        raise ValueError("invalid tensor-parallel group size")
    if world_size == 1:
        return bool(failed)
    import torch
    from vllm.distributed.parallel_state import get_tp_group
    group = get_tp_group()
    if group.world_size != world_size:
        raise RuntimeError("Drift tensor-parallel group size mismatch")
    status = torch.tensor([int(bool(failed))], dtype=torch.int32, device="cpu")
    torch.distributed.all_reduce(status, op=torch.distributed.ReduceOp.MAX, group=group.cpu_group)
    return bool(status.item())
