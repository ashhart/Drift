"""Validate completion SSE terminal evidence without exposing response contents."""
import json


class CompletionStream:
    def __init__(self, maximum_bytes=1_048_576):
        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise ValueError('GLM_COMPLETION_STREAM')
        self.maximum, self.bytes = maximum_bytes, 0
        self.reason, self.usage, self.done = None, None, False

    def accept(self, raw):
        try:
            return self._accept(raw)
        except (ValueError, TypeError, KeyError, AttributeError):
            raise ValueError('GLM_COMPLETION_STREAM') from None

    def _accept(self, raw):
        self.bytes += len(raw)
        if self.done or self.bytes > self.maximum:
            raise ValueError
        line = raw.decode('utf-8').strip()
        if not line or line.startswith(':'):
            return None
        if not line.startswith('data:'):
            raise ValueError
        payload = line[5:].strip()
        if payload == '[DONE]':
            self.done = True
            return None
        event = json.loads(payload)
        if type(event) is not dict or 'error' in event:
            raise ValueError
        choices = event.get('choices')
        if type(choices) is not list or len(choices) > 1:
            raise ValueError
        usage = event.get('usage')
        if usage is not None:
            if self.usage is not None or type(usage) is not dict:
                raise ValueError
            counts = [usage.get('prompt_tokens'), usage.get('completion_tokens')]
            if any(type(value) is not int or value < 0 for value in counts):
                raise ValueError
            self.usage = dict(zip(('prompt_tokens', 'completion_tokens'), counts))
            if 'total_tokens' in usage:
                total = usage['total_tokens']
                if type(total) is not int or total != sum(counts):
                    raise ValueError
                self.usage['total_tokens'] = total
        if not choices:
            return None
        choice = choices[0]
        if type(choice) is not dict or self.reason is not None:
            raise ValueError
        text, reason = choice.get('text'), choice.get('finish_reason')
        if type(text) is not str or reason not in (None, 'stop', 'length'):
            raise ValueError
        self.reason = reason
        return text or None

    def finish(self, *, prompt_tokens, max_tokens):
        if not self.done or self.reason is None or self.usage is None:
            raise ValueError('GLM_COMPLETION_STREAM')
        if self.usage['prompt_tokens'] != prompt_tokens or self.usage['completion_tokens'] > max_tokens:
            raise ValueError('GLM_COMPLETION_STREAM')
        return {'finish_reason': self.reason, 'usage': dict(self.usage), 'stream_done': True}
