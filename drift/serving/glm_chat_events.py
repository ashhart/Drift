"""Decode bounded native chat-completion deltas without guessing tool syntax."""
import json


class ChatEvents:
    def __init__(self, tools, maximum_bytes):
        self.tools, self.maximum = set(tools), maximum_bytes
        self.calls, self.content, self.reason, self.usage = {}, '', None, None
        self.bytes = 0

    def accept(self, event):
        self.bytes += len(json.dumps(event, allow_nan=False).encode())
        if self.bytes > self.maximum or type(event) is not dict:
            raise ValueError('chat stream exceeds its bound')
        choices = event.get('choices')
        if type(choices) is not list or len(choices) > 1 or 'error' in event:
            raise ValueError('invalid chat stream')
        usage = event.get('usage')
        if usage is not None:
            if self.usage is not None or type(usage) is not dict:
                raise ValueError('duplicate or malformed usage')
            values = [usage.get('prompt_tokens'), usage.get('completion_tokens')]
            if any(type(value) is not int or value < 0 for value in values):
                raise ValueError('missing exact usage')
            self.usage = {'input_tokens': values[0], 'output_tokens': values[1]}
        if not choices:
            return []
        choice = choices[0]
        if type(choice) is not dict or self.reason is not None:
            raise ValueError('data after terminal choice')
        delta = choice.get('delta')
        if type(delta) is not dict:
            raise ValueError('invalid delta')
        content = delta.get('content')
        output = []
        if content is not None:
            if type(content) is not str:
                raise ValueError('invalid text delta')
            self.content += content
            if content:
                output.append({'op': 'text', 'payload': {'text': content}})
        for call in delta.get('tool_calls') or []:
            self._call(call)
        reason = choice.get('finish_reason')
        if reason is not None:
            if reason not in ('stop', 'tool_calls', 'length'):
                raise ValueError('unsupported completion reason')
            self.reason = reason
        return output

    def _call(self, fragment):
        index = fragment.get('index')
        if type(index) is not int or not 0 <= index < 16:
            raise ValueError('invalid tool index')
        target = self.calls.setdefault(index, {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
        if fragment.get('type', 'function') != 'function':
            raise ValueError('unsupported tool type')
        if fragment.get('id'):
            if target['id']:
                raise ValueError('repeated tool identity')
            target['id'] = fragment['id']
        function = fragment.get('function') or {}
        for key in ('name', 'arguments'):
            part = function.get(key)
            if part is not None:
                if type(part) is not str:
                    raise ValueError('invalid function delta')
                target['function'][key] += part

    def finish(self):
        if self.reason is None or self.usage is None:
            raise ValueError('chat stream missing terminal evidence')
        if bool(self.calls) != (self.reason == 'tool_calls'):
            raise ValueError('tool calls disagree with terminal reason')
        calls, identities, output = [], set(), []
        for index, call in sorted(self.calls.items()):
            function = call['function']
            if index != len(calls) or type(call['id']) is not str or not 0 < len(call['id']) <= 128 or call['id'] in identities:
                raise ValueError('invalid tool identity')
            if function['name'] not in self.tools:
                raise ValueError('unknown tool')
            arguments = json.loads(function['arguments'])
            if type(arguments) is not dict:
                raise ValueError('tool arguments must be an object')
            identities.add(call['id']); calls.append(call)
            output.append({'op': 'tool_call', 'payload': {'call_id': call['id'], 'name': function['name'], 'arguments': arguments}})
        message = {'role': 'assistant', 'content': self.content}
        if calls:
            message['tool_calls'] = calls
        reason = 'tool_use' if calls else self.reason
        output.append({'op': 'terminal', 'payload': {'reason': reason, 'usage': self.usage}})
        return message, identities, output
