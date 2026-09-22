"""Convert completed text-only chat messages; never collect credentials or hidden reasoning."""
import json

from drift.transcript.schema import Transcript, fields, require


def from_chat_messages(messages, *, provider, model, conversation_id):
    require(type(messages) is list and 0 < len(messages) <= 512)
    converted = []
    for index, item in enumerate(messages):
        fields(item, ('role', 'content'), ('tool_calls', 'tool_call_id'))
        row = {'id': f'm{index + 1}', 'source': 'api', 'role': item['role'], 'content': item['content']}
        if item['content'] is None and item['role'] == 'assistant' and item.get('tool_calls'):
            row['content'] = ''
        if 'tool_call_id' in item:
            row['tool_call_id'] = item['tool_call_id']
        if 'tool_calls' in item:
            require(type(item['tool_calls']) is list)
            row['tool_calls'] = []
            for call in item['tool_calls']:
                fields(call, ('id', 'type', 'function'))
                require(call['type'] == 'function')
                fields(call['function'], ('name', 'arguments'))
                row['tool_calls'].append({'id': call['id'], **call['function']})
        converted.append(row)
    return Transcript.parse({'schema': 'drift.transcript.v1', 'conversation_id': conversation_id,
                             'sources': {'api': {'provider': provider, 'model': model}}, 'messages': converted})


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)
