"""Deterministic actor for the installed OMP KV-only lifecycle check."""
import json
from pathlib import Path


class KvFixture:
    backend = 'fixture'
    capabilities = {'native_state': 'fixture', 'tool_calls': True, 'cancellation': True}

    def __init__(self, config):
        self.path = Path(config['fixture_report'])
        self.role = config['fixture_role']
        self.facts = dict(open=0, own_prompt=0, stream=0, tool_result=0, close=0)

    def record(self, key):
        self.facts[key] += 1
        self.path.write_text(json.dumps(self.facts))

    def open(self, payload):
        self.record('open')

    def own_prompt(self, payload):
        self.record('own_prompt')

    def tool_result(self, payload):
        if payload['text'] != 'ready' or payload['is_error']:
            raise ValueError('FIXTURE_TEXT_CHANNEL')
        self.record('tool_result')

    def stream(self, max_tokens):
        self.record('stream')
        turn = self.facts['stream']
        if self.role == 'parent':
            name = 'task' if turn == 1 else 'drift_sync' if turn <= 3 else None
        else:
            name = 'drift_sync' if turn <= 2 else 'yield'
        if name:
            yield dict(op='tool_call', payload=dict(call_id=f'{self.role}-{turn}', name=name, arguments={}))
        else:
            yield dict(op='text', payload=dict(text='Own answer, never peer input.'))
        yield dict(op='terminal', payload=dict(reason='tool_use' if name else 'stop', usage=dict(input_tokens=3, output_tokens=2)))

    def cancel(self, seq):
        pass

    def close(self):
        if not self.facts['close']:
            self.record('close')
