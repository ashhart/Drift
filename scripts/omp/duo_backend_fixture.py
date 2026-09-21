"""Drive stock Duo tools through the actual worker core with deterministic outputs."""
import hashlib
import time
from scripts.omp.worker_backend_fixture import LifetimeFixture


class DuoFixture(LifetimeFixture):
    def __init__(self, config):
        super().__init__(config)
        self.worker = config['worker']
        self.tools = {}
        self.facts['errors'] = 0

    def open(self, payload):
        self.tools = {tool['name']: tool['parameters'] for tool in payload['tools']}
        self.facts['advertised_tools'] = sorted(self.tools)
        self.facts['has_task'] = 'task' in self.tools
        self.facts['has_todo'] = 'todo' in self.tools
        self.record('open')

    def own_prompt(self, payload):
        self.facts['own_prompt_sha256'] = hashlib.sha256(payload['text'].encode()).hexdigest()
        self.record('own_prompt')

    def tool_result(self, payload):
        self.skipped = payload['is_error'] and payload['text'].startswith('Skipped due to pending system advisory.')
        self.facts['errors'] += int(payload['is_error'] and not self.skipped)
        if self.skipped:
            self.facts['skipped'] = self.facts.get('skipped', 0) + 1
        self.record('tool_result')

    def stream(self, maximum):
        self.record('stream')
        turn = self.facts['stream']
        call = None
        if self.worker == 'glm-fixture':
            if turn == 1:
                call = ('todo', {'op': 'view'})
            elif turn == 2:
                item = {'agent': 'duo-peer', 'name': 'DuoPeer', 'task': 'Child fixture task: send PROPOSE fixture through Hub, then return fixture completion.'}
                properties = self.tools['task'].get('properties', {})
                call = ('task', {'context': 'Synthetic Duo fixture', 'tasks': [item]} if 'tasks' in properties else item)
            elif turn == 3 or getattr(self, 'skipped', False):
                call = ('hub', {'op': 'wait', 'ids': ['DuoPeer'], 'timeoutMs': 2000})
        elif turn == 1:
            call = ('hub', {'op': 'send', 'to': 'Main', 'message': 'PROPOSE fixture task split'})
        if call:
            yield {'op': 'tool_call', 'payload': {'call_id': self.worker + '-' + str(turn), 'name': call[0], 'arguments': dict(call[1], i='fixture intent')}}
        else:
            if self.worker == 'glm-fixture':
                time.sleep(0.2)
            yield {'op': 'text', 'payload': {'text': 'Fixture complete'}}
        yield {'op': 'terminal', 'payload': {'reason': 'tool_use' if call else 'stop', 'usage': {'input_tokens': 3, 'output_tokens': 2}}}
