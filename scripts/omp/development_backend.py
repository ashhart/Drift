"""Exercise the real scoped API tools using synthetic worker events, never model inference."""
from scripts.omp.duo_backend_fixture import DuoFixture
from scripts.omp.api_fixture import API


class DevelopmentFixture(DuoFixture):
    def tool_result(self, payload):
        super().tool_result(payload)
        if payload['call_id'].startswith('api-') and not payload['is_error']:
            self.api_step = getattr(self, 'api_step', 0) + 1
        self.facts['api_step'] = getattr(self, 'api_step', 0)
        self.facts.setdefault('api_result', 0)
        self.record('api_result')

    def stream(self, maximum):
        turn = getattr(self, 'api_step', 0)
        if self.worker != 'qwen-fixture' or turn >= 5:
            yield from super().stream(maximum)
            return
        self.record('stream')
        calls = [('drift_task_read', {'path': 'api.py'}), ('drift_task_write', {'path': 'api.py', 'text': API.replace('(500, {"status": "broken"})', '(200, {"status": "ok"})')}), ('drift_task_verify', {}), ('hub', {'op': 'send', 'to': 'Main', 'message': 'PROPOSE public API fixture implemented and checked'}), ('yield', {'result': {'data': {'summary': 'Public API checks passed'}}})]
        name, arguments = calls[turn]
        if name == 'yield' and 'data' in self.tools['yield'].get('properties', {}): arguments = {'data': {'summary': 'Public API checks passed'}}
        yield {'op': 'tool_call', 'payload': {'call_id': f"api-{self.facts['stream']}", 'name': name, 'arguments': arguments}}
        yield {'op': 'terminal', 'payload': {'reason': 'tool_use', 'usage': {'input_tokens': 3, 'output_tokens': 2}}}
