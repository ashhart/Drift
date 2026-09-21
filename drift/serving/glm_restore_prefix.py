"""Own the cancellable private tokenizer-proof child without returning prompt token IDs."""
from pathlib import Path
from drift.serving.glm_http import GlmHttp


class PrefixVerifier(GlmHttp):
    def _script(self, path):
        if path != '/verify-reservation':
            raise ValueError('prefix verifier endpoint mismatch')
        return Path(__file__).with_name('glm_restore_prefix_child.py')

    def verify(self, original, reserved, rows, timeout):
        frames = list(self._frames('/verify-reservation', {'original': original, 'reserved': reserved, 'rows': rows}, timeout))
        expected = {'verified', 'own_tokens', 'prompt_tokens', 'reserve_start', 'reserve_tokens'}
        if len(frames) != 1 or set(frames[0]) != expected or frames[0]['verified'] is not True:
            raise ValueError('invalid native tokenizer proof')
        proof = frames[0]
        if any(type(proof[key]) is not int or proof[key] < 0 for key in expected - {'verified'}):
            raise ValueError('invalid native tokenizer proof counters')
        if proof['reserve_tokens'] != rows or proof['prompt_tokens'] != proof['own_tokens'] + rows or proof['reserve_start'] >= proof['own_tokens']:
            raise ValueError('native tokenizer proof layout mismatch')
        return proof
