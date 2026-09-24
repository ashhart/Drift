"""Use the installed async scheduler while ending a linked request's initial prefill at its boundary."""
from vllm.v1.core.sched.async_scheduler import AsyncScheduler

try:
    from glm_prefill_boundary import split
except ImportError:
    from drift.serving.glm_prefill_boundary import split


class DriftScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.need_mamba_block_aligned_split or not self.scheduler_config.async_scheduling:
            raise ValueError('Drift scheduler requires the align-mode GLM cache with async scheduling')

    def _mamba_block_aligned_split(self, request, num_new_tokens,
                                   num_new_local_computed_tokens=0, num_external_computed_tokens=0):
        start = request.num_computed_tokens+num_new_local_computed_tokens+num_external_computed_tokens
        parent = super()._mamba_block_aligned_split
        result = split(request, num_new_tokens, start, self.cache_config.block_size,
                       lambda count: parent(request, count, num_new_local_computed_tokens,
                                            num_external_computed_tokens))
        if type(result) is not int or not 0 <= result <= num_new_tokens:
            raise ValueError('prefill split exceeded the scheduled tokens')
        return result
