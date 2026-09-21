"""Exercise the rank-failure vote with two local CPU processes, without vLLM."""
import multiprocessing

import pytest


def vote_worker(rank, failing_rank, rendezvous, output):
    import sys
    from datetime import timedelta
    from types import ModuleType, SimpleNamespace
    import torch.distributed as distributed
    from drift.serving.glm_rank_failure import any_rank_failed
    distributed.init_process_group("gloo", init_method=rendezvous, rank=rank,
                                   world_size=2, timeout=timedelta(seconds=10))
    try:
        module = ModuleType("vllm.distributed.parallel_state")
        module.get_tp_group = lambda: SimpleNamespace(world_size=2, cpu_group=distributed.group.WORLD)
        sys.modules[module.__name__] = module
        output.send(any_rank_failed(rank == failing_rank, 2))
    finally:
        distributed.destroy_process_group()
        output.close()


@pytest.mark.parametrize("failing_rank", [None, 0, 1])
def test_both_ranks_observe_either_ranks_failure(tmp_path, failing_rank):
    context = multiprocessing.get_context("spawn")
    workers, pipes = [], []
    try:
        for rank in range(2):
            read, write = context.Pipe(duplex=False)
            worker = context.Process(target=vote_worker, args=(
                rank, failing_rank, (tmp_path / "rendezvous").as_uri(), write))
            worker.start()
            write.close()
            workers.append(worker)
            pipes.append(read)
        for pipe in pipes:
            assert pipe.poll(15)
            assert pipe.recv() is (failing_rank is not None)
        for worker in workers:
            worker.join(10)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(10)
        for pipe in pipes:
            pipe.close()
