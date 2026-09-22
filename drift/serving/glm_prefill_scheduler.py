"""Use the installed async scheduler while capping explicitly guarded prefill."""
from vllm.v1.core.sched.async_scheduler import AsyncScheduler

try:
    from glm_prefill_boundary import cap_prefill
except ImportError:
    from drift.serving.glm_prefill_boundary import cap_prefill


class DriftScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.need_mamba_block_aligned_split or self.cache_config.block_size != 64:
            raise ValueError('Drift scheduler requires the qualified aligned GLM cache')

    def _mamba_block_aligned_split(self, request, num_new_tokens,
                                   num_new_local_computed_tokens=0, num_external_computed_tokens=0):
        start = request.num_computed_tokens+num_new_local_computed_tokens+num_external_computed_tokens
        cap = cap_prefill(request, num_new_tokens, start, self.cache_config.block_size)
        result = super()._mamba_block_aligned_split(request, cap, num_new_local_computed_tokens,
                                                   num_external_computed_tokens)
        if type(result) is not int or not 0 <= result <= cap:
            raise ValueError('native scheduler exceeded the guarded prefill cap')
        return result
