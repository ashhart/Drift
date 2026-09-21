"""Deterministic backend for the actual WorkerSession and stdio implementation."""
import json
from pathlib import Path


class FixtureBackend:
    backend = 'fixture'
    capabilities = {'native_state': 'fixture', 'tool_calls': True, 'cancellation': True}

    def __init__(self, config):
        self.path = Path(config['fixture_report'])
        self.facts = dict(open=0, own_prompt=0, tool_result=0, stream=0, close=0)

    def record(self, operation):
        self.facts[operation] += 1
        self.path.write_text(json.dumps(self.facts))

    def open(self, payload):
        self.record('open')

    def own_prompt(self, payload):
        self.record('own_prompt')

    def tool_result(self, payload):
        assert payload == dict(call_id='fixture-call', text='fixture-ok', is_error=False)
        self.record('tool_result')

    def stream(self, max_tokens):
        self.record('stream')
        first = self.facts['stream'] == 1
        for text in (['fixture', ' ', 'step', '\n'] if first else ['fixture-complete']):
            yield dict(op='text', payload=dict(text=text))
        if first:
            yield dict(op='tool_call', payload=dict(call_id='fixture-call', name='drift_contract_echo', arguments={'i': 'fixture intent'}))
        yield dict(op='terminal', payload=dict(reason='tool_use' if first else 'stop', usage=dict(input_tokens=3, output_tokens=2)))

    def cancel(self, seq):
        pass

    def close(self):
        if not self.facts['close']:
            self.record('close')


class LifetimeFixture(FixtureBackend):
    def own_control_append(self, payload):
        self.facts['own_control_append'] = self.facts.get('own_control_append', 0) + 1
        self.path.write_text(json.dumps(self.facts))

    def own_control(self, payload):
        self.facts['own_control'] = self.facts.get('own_control', 0) + 1
        self.path.write_text(json.dumps(self.facts))

    def stream(self, max_tokens):
        self.record('stream')
        yield dict(op='text', payload=dict(text='fixture-complete'))
        yield dict(op='terminal', payload=dict(reason='stop', usage=dict(input_tokens=3, output_tokens=2)))
