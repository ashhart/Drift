"""Retain raw assistant markup while detecting changes to its parsed semantic history."""
from copy import deepcopy
import json
from drift.serving.worker_contract import require


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


class AssistantHistory:
    def __init__(self): self.entries = {}

    def remember(self, index, message, raw_text):
        require(type(raw_text) is str and index not in self.entries)
        self.entries[index] = (canonical(message), raw_text)

    def render(self, messages):
        rendered = deepcopy(messages)
        for index, (semantic, raw) in self.entries.items():
            require(index < len(messages) and canonical(messages[index]) == semantic, 'CAPABILITY')
            rendered[index] = {'role': 'assistant', 'content': raw}
        return rendered
