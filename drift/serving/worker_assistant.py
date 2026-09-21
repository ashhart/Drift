"""Normalize parsed assistant tool semantics independently of native serialized text."""
from copy import deepcopy
import json
from drift.serving.worker_contract import require


def parsed_assistant(cleaned, calls):
    require(type(cleaned) is str and (calls is None or type(calls) in (list, tuple)))
    message, events, normalized = {'role': 'assistant', 'content': cleaned}, [], []
    for call in calls or []:
        value = deepcopy(call.model_dump())
        require(type(value) is dict and value.get('type') == 'function')
        function = value.get('function'); require(type(function) is dict)
        arguments = function.get('arguments')
        if type(arguments) is str: arguments = json.loads(arguments)
        require(type(arguments) is dict and type(value.get('id')) is str and type(function.get('name')) is str)
        json.dumps(arguments, allow_nan=False)
        function['arguments'] = arguments
        normalized.append(value)
        events.append({'call_id': value['id'], 'name': function['name'], 'arguments': deepcopy(arguments)})
    if normalized: message['tool_calls'] = normalized
    return message, events
