import json
from copy import deepcopy
import pytest
from drift.serving.worker_codec import CheckpointCodec


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        text = ''
        for message in messages:
            content = message['content']
            if message.get('tool_calls'):
                content = '<tool_call>' + json.dumps(message['tool_calls'][0]['function']['arguments'], sort_keys=True) + '</tool_call>'
            text += '<' + message['role'] + '>' + content + '~'
        if kwargs.get('add_generation_prompt'): text += '<assistant>'
        return list(text.encode())
    def decode(self, ids, **kwargs): return bytes(ids).decode()


def prepared():
    codec = CheckpointCodec(Tokenizer(), ['original system'], [])
    codec.append({'role': 'user', 'content': 'own question'})
    return codec


@pytest.mark.parametrize('raw', ['<tool_call>{"value": "a"}</tool_call>', '<tool_call>{ "value" : "a" }\n \n</tool_call>'])
def test_native_assistant_bytes_survive_template_canonicalization(raw):
    codec = prepared()
    message = {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call', 'type': 'function', 'function': {'name': 'echo', 'arguments': {'value': 'a'}}}]}
    codec.assistant(message, list(raw.encode()), ord('~'), raw_text=raw)
    consumed = list(codec.consumed)
    suffix = codec.append({'role': 'tool', 'tool_call_id': 'call', 'content': 'a'})
    assert codec.consumed[:len(consumed)] == consumed and suffix
    assert codec.messages[-2] == message


@pytest.mark.parametrize('tamper', ['semantic', 'own', 'raw'])
def test_native_assistant_snapshot_and_own_input_guards_remain_strict(tamper):
    codec = prepared(); raw = 'exact answer'
    message = {'role': 'assistant', 'content': raw}
    if tamper == 'raw':
        with pytest.raises(RuntimeError): codec.assistant(message, list(raw.encode()), ord('~'), raw_text='different')
        return
    codec.assistant(message, list(raw.encode()), ord('~'), raw_text=raw)
    if tamper == 'semantic': codec.messages[-1]['content'] = 'changed'
    else: codec.messages[0]['content'] = 'changed'
    with pytest.raises(RuntimeError, match='CAPABILITY'): codec.append({'role': 'user', 'content': 'next'})


def test_native_parser_string_arguments_are_normalized_without_semantic_changes():
    from drift.serving.worker_assistant import parsed_assistant
    class Call:
        def model_dump(self): return {'id': 'call', 'type': 'function', 'function': {'name': 'echo', 'arguments': '{ "i":"intent", "value": [1, true, "x"] }'}}
    message, events = parsed_assistant('', [Call()])
    expected = {'i': 'intent', 'value': [1, True, 'x']}
    assert message['tool_calls'][0]['function']['arguments'] == expected
    assert events[0]['arguments'] == expected


def test_recorded_stop_boundary_is_still_part_of_the_exact_prefix():
    codec = prepared(); raw = 'exact answer'
    codec.assistant({'role': 'assistant', 'content': raw}, list(raw.encode()), ord('!'), raw_text=raw)
    with pytest.raises(RuntimeError, match='CAPABILITY'):
        codec.append({'role': 'user', 'content': 'next'})


@pytest.mark.parametrize('arguments', ['[]', 'null', '{"x": NaN}'])
def test_native_parser_arguments_must_be_a_finite_object(arguments):
    from drift.serving.worker_assistant import parsed_assistant
    class Call:
        def model_dump(self): return {'id': 'call', 'type': 'function', 'function': {'name': 'echo', 'arguments': arguments}}
    with pytest.raises((ValueError, RuntimeError)): parsed_assistant('', [Call()])
