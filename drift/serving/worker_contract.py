"""Validate bounded private worker envelopes without admitting activation transport data."""
import json
import re


class WorkerError(RuntimeError):
    pass


def require(condition, code='PROTOCOL'):
    if not condition:
        raise WorkerError(code)


def size(value):
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8'))


def integer(value, low, high, code='LIMIT'):
    require(type(value) is int and low <= value <= high, code)
    return value


def fields(value, names):
    require(type(value) is dict and set(value) == set(names))


def validate_open(payload, pins, ceilings):
    fields(payload, (*pins, 'limits', 'system_prompt', 'tools'))
    require(all(payload[key] == value for key, value in pins.items()), 'CAPABILITY')
    fields(payload['limits'], ceilings)
    for name, ceiling in ceilings.items():
        integer(payload['limits'][name], 1, ceiling)
    require(type(payload['system_prompt']) is list and all(type(x) is str for x in payload['system_prompt']))
    require(type(payload['tools']) is list)
    names = set()
    for tool in payload['tools']:
        fields(tool, ('name', 'description', 'parameters'))
        require(type(tool['name']) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,96}', tool['name']) is not None)
        require(tool['name'] not in names and type(tool['description']) is str and type(tool['parameters']) is dict)
        names.add(tool['name'])
    require(size(payload) <= payload['limits']['max_input_bytes'], 'LIMIT')


def validate_event(event, tools):
    fields(event, ('op', 'payload'))
    op, payload = event['op'], event['payload']
    if op == 'text':
        fields(payload, ('text',)); require(type(payload['text']) is str)
    elif op == 'tool_call':
        fields(payload, ('call_id', 'name', 'arguments'))
        require(type(payload['call_id']) is str and 0 < len(payload['call_id']) <= 128)
        require(payload['name'] in tools and type(payload['arguments']) is dict)
    elif op == 'terminal':
        fields(payload, ('reason', 'usage'))
        require(payload['reason'] in ('stop', 'tool_use', 'length'))
        fields(payload['usage'], ('input_tokens', 'output_tokens'))
    else:
        raise WorkerError('PROTOCOL')
