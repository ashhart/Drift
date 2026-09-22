"""Prepare one fresh restored native-chat turn and gate output on exact rank receipts."""
import copy
import json
import os
import re
from pathlib import Path
import time
import uuid
from drift.serving.glm_restore_prompt import reserve_prompt
from drift.serving.glm_restore_padding import padded_reservation
from drift.exchange.lifetime import request_scope


class TurnRestorer:
    def __init__(self, verifier, publication, *, rows, digest, root, deadline, max_input_bytes, linked=True, clock=time.monotonic, outbox=None, causal_prefill=False):
        if type(rows) is not int or not 1 <= rows <= 4096 or type(linked) is not bool or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('invalid restoration recipe')
        self.outbox = outbox
        if type(causal_prefill) is not bool: raise ValueError('invalid causal prefill policy')
        self.causal_prefill = causal_prefill
        if outbox is not None and not linked: raise ValueError('outbox requires explicit linked native session')
        self.verifier, self.publication = verifier, publication
        self.rows, self.digest, self.root = rows, digest, Path(root)
        self.deadline, self.max_input_bytes, self.linked, self.clock = deadline, max_input_bytes, linked, clock
        self.plan, self.failed, self.completed = None, False, []

    def remaining(self):
        value = self.deadline() - self.clock()
        if self.failed or value <= 0:
            raise TimeoutError('restoration session unavailable')
        return value

    def prepare(self, body):
        self.remaining()
        if self.plan is not None: raise ValueError('previous restoration turn is unfinished')
        reserved = reserve_prompt(body, self.rows)
        if len(json.dumps(reserved).encode()) > self.max_input_bytes:
            raise ValueError('reserved own input exceeds byte budget')
        reserved, proof = padded_reservation(self.verifier, body, self.rows, self.remaining, self.causal_prefill)
        if len(json.dumps(reserved).encode()) > self.max_input_bytes:
            raise ValueError('padded own input exceeds byte budget')
        self.remaining()
        name = 'restore-' + uuid.uuid4().hex
        extra = {'cache_salt': 'drift:' + name, 'vllm_xargs': {'skip_writing_prefix_cache': 1}}
        publication = None
        if self.linked:
            extra['kv_transfer_params'] = {'drift_session': name, 'drift_reserve': self.rows,
                                           'drift_reserve_start': proof['reserve_start'], 'drift_tap': self.outbox is not None}
            if self.causal_prefill:
                extra['kv_transfer_params']['drift_prefill_boundary'] = proof['prefill_boundary']
        elif self.causal_prefill:
            extra['kv_transfer_params'] = {'drift_no_link': True, 'drift_reserve': self.rows,
                                           'drift_reserve_start': proof['reserve_start'],
                                           'drift_prefill_boundary': proof['prefill_boundary']}
        if len(json.dumps({**reserved, **extra}).encode()) > self.max_input_bytes:
            raise ValueError('prepared input exceeds byte budget')
        if self.outbox is not None:
            with request_scope(self.remaining):
                self.outbox.begin(name, proof, body['max_tokens'])
        if self.linked:
            if self.root != self.root.resolve() or not self.root.is_dir():
                raise ValueError('private memory root differs')
            for kind in ('tp-live-in', 'tp-live-out'):
                parent = self.root / kind
                parent.mkdir(mode=0o700, exist_ok=True)
                if parent.is_symlink() or parent.stat().st_uid != os.getuid() or parent.stat().st_mode & 0o022:
                    raise ValueError('unsafe private memory parent')
                (parent / name).mkdir(mode=0o700)
            publication = self.publication(name)
            publication.delivery.health(deadline=self.deadline())
            publication.stage(timeout=self.remaining())
            publication.delivery.admit(0, self.digest, deadline=self.deadline())
            publication.publish(timeout=self.remaining())
        self.remaining()
        self.plan = {'body': copy.deepcopy(body), 'extra': extra, 'proof': proof, 'publication': publication,
                     'name': name, 'counted': False, 'started': False, 'receipts': None}
        return copy.deepcopy(extra)

    def request(self, body):
        self.remaining()
        if self.plan is None: raise ValueError('restoration not prepared')
        expected = {**self.plan['body'], **self.plan['extra']}
        maximum = body.get('max_tokens')
        if type(maximum) is not int or not 1 <= maximum <= expected['max_tokens']:
            raise ValueError('restored output budget changed')
        expected['max_tokens'] = maximum
        if body != expected: raise ValueError('prepared own input changed')
        return reserve_prompt(body, self.rows+self.plan['proof'].get('padding_tokens', 0))

    def applied(self):
        self.remaining()
        plan = self.plan
        if plan['receipts'] is None:
            plan['receipts'] = plan['publication'].delivery.wait_applied(0, self.rows, self.digest, deadline=self.deadline()) if self.linked else []
            self.remaining()

    def finish(self):
        self.applied()
        outbound = self.outbox.finish(self.plan['name'], self.plan['max_tokens'], self.deadline()) if self.outbox is not None else None
        self.completed.append({'session': self.plan['name'], **self.plan['proof'], 'linked': self.linked,
                               'snapshot_sha256': self.digest, 'receipts': self.plan['receipts'],
                               'first_token_causality': 'NOT_ESTABLISHED',
                               'prefill_policy': 'GUARDED_REQUESTED' if self.causal_prefill else 'UNGUARDED'})
        if outbound is not None: self.completed[-1]['outbox'] = outbound
        self.plan = None

    def close(self):
        self.failed = True
        self.verifier.close()
        if self.outbox is not None: self.outbox.close()

    def report(self):
        return {'completed_turns': len(self.completed), 'turns': copy.deepcopy(self.completed), 'failed': self.failed}


class RestoringTransport:
    def __init__(self, inner, restorer):
        self.inner, self.restorer = inner, restorer

    def count_tokens(self, body, timeout):
        request = self.restorer.request(body)
        count = self.inner.count_tokens(request, min(timeout, self.restorer.remaining()))
        self.restorer.remaining()
        if count != self.restorer.plan['proof']['prompt_tokens']:
            raise ValueError('reserved prompt token count changed')
        self.restorer.plan['counted'] = True
        return count

    def stream(self, body, timeout):
        try:
            request = self.restorer.request(body)
            if not self.restorer.plan['counted'] or self.restorer.plan['started']:
                raise ValueError('restoration request not admitted or already used')
            self.restorer.plan['started'] = True
            self.restorer.plan['max_tokens'] = body['max_tokens']
            for event in self.inner.stream(request, min(timeout, self.restorer.remaining())):
                self.restorer.applied()
                yield event
            self.restorer.finish()
        except BaseException:
            self.close()
            raise

    def close(self):
        self.restorer.close()
        self.inner.close()

    def cancel(self):
        self.restorer.close()
        self.inner.cancel()
