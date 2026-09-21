"""Replay the actual Spark session tail against synthetic completion SSE, without HTTP or models."""
import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]


def data(value):
    return b'data: ' + json.dumps(value).encode() + b'\n\n'


def transcript(reason='stop', *, footer=True, done=True):
    frames = [data({'choices': [{'text': 'synthetic answer', 'finish_reason': None}]}),
              data({'choices': [{'text': '', 'finish_reason': reason}]})]
    if footer:
        frames.append(data({'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13}}))
    if done:
        frames.append(b'data: [DONE]\n\n')
    return b''.join(frames)


def replay(tmp_path, content):
    tree = ast.parse((ROOT / 'scripts/mcdma_target/spark_glm_session.py').read_text())
    start = next(index for index, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                 and any(isinstance(item, ast.Name) and item.id == 't0' for item in ast.walk(node.targets[0])))
    namespace = dict(json=json, time=time, ROOT=tmp_path, prompt=list(range(10)), body={},
                     args=SimpleNamespace(tap=True, session='synthetic-session', max_new=8),
                     post=lambda *args: io.BytesIO(content))
    spec = importlib.util.find_spec('scripts.mcdma_target.glm_completion_sse')
    if spec is not None:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        namespace['CompletionStream'] = module.CompletionStream
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(ast.Module(body=tree.body[start:], type_ignores=[]), '<actual-spark-session-tail>', 'exec'), namespace)
    return [json.loads(line) for line in output.getvalue().splitlines()]


@pytest.mark.parametrize('reason', ['stop', 'length'])
def test_actual_session_preserves_terminal_reason_and_usage_footer(tmp_path, reason):
    stream, answer = replay(tmp_path, transcript(reason))
    assert answer['text'] == 'synthetic answer'
    assert answer['finish_reason'] == reason
    assert answer['usage'] == {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13}
    assert answer['stream_done'] is True
    assert len(stream['stream']['chunks']) == 1


@pytest.mark.parametrize('content', [transcript(done=False), transcript(footer=False),
                                     data({'choices': [{'text': 'synthetic answer', 'finish_reason': None}]})])
def test_actual_session_rejects_missing_terminal_evidence(tmp_path, content):
    with pytest.raises(ValueError, match='GLM_COMPLETION_STREAM'):
        replay(tmp_path, content)


@pytest.mark.parametrize('event', [
    {'error': {'message': 'private response must not escape'}, 'choices': []},
    {'choices': [{'text': 'private response', 'finish_reason': 'unknown'}]},
    {'choices': [{'text': 123, 'finish_reason': None}]},
    {'choices': [], 'usage': {'prompt_tokens': True, 'completion_tokens': 3}},
    {'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': -1}},
    {'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 99}},
    {'choices': {}, 'private': 'response'},
])
def test_malformed_events_fail_without_response_contents(tmp_path, event):
    with pytest.raises(ValueError) as caught:
        replay(tmp_path, data(event))
    assert str(caught.value) == 'GLM_COMPLETION_STREAM'
    assert caught.value.__suppress_context__


@pytest.mark.parametrize('raw', [b'data: private malformed json\n', b'data: \xff\n', b'private invalid SSE\n'])
def test_malformed_wire_data_is_sanitized(tmp_path, raw):
    with pytest.raises(ValueError, match='^GLM_COMPLETION_STREAM$'):
        replay(tmp_path, raw)


@pytest.mark.parametrize('prompt_tokens, completion_tokens', [(11, 3), (10, 9)])
def test_usage_must_match_prompt_and_generation_budget(tmp_path, prompt_tokens, completion_tokens):
    footer = data({'choices': [], 'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens}})
    with pytest.raises(ValueError, match='^GLM_COMPLETION_STREAM$'):
        replay(tmp_path, transcript(footer=False, done=False) + footer + b'data: [DONE]\n')


def test_terminal_text_and_usage_in_same_event_are_preserved(tmp_path):
    event = {'choices': [{'text': 'last piece', 'finish_reason': 'length'}],
             'usage': {'prompt_tokens': 10, 'completion_tokens': 8}}
    stream, answer = replay(tmp_path, b': keepalive\n\n' + data(event) + b'data: [DONE]\n')
    assert answer['text'] == 'last piece'
    assert answer['usage'] == event['usage']
    assert answer['finish_reason'] == 'length'
    assert len(stream['stream']['chunks']) == 1


@pytest.mark.parametrize('extra', [data({'choices': [{'text': '', 'finish_reason': 'stop'}]}),
                                 data({'choices': [{'text': 'late private text', 'finish_reason': None}]}),
                                 data({'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 3}})])
def test_duplicate_terminal_or_usage_and_post_terminal_text_are_rejected(tmp_path, extra):
    with pytest.raises(ValueError, match='^GLM_COMPLETION_STREAM$'):
        replay(tmp_path, transcript(done=False) + extra + b'data: [DONE]\n')


def test_parser_enforces_byte_bound_and_rejects_data_after_done():
    from scripts.mcdma_target.glm_completion_sse import CompletionStream
    with pytest.raises(ValueError, match='^GLM_COMPLETION_STREAM$'):
        CompletionStream(maximum_bytes=4).accept(b'data: [DONE]\n')
    parser = CompletionStream()
    parser.accept(b'data: [DONE]\n')
    with pytest.raises(ValueError, match='^GLM_COMPLETION_STREAM$'):
        parser.accept(b'\n')


def test_empty_completion_has_no_first_token_timestamp(tmp_path):
    event = {'choices': [{'text': '', 'finish_reason': 'stop'}],
             'usage': {'prompt_tokens': 10, 'completion_tokens': 0}}
    stream, answer = replay(tmp_path, data(event) + b'data: [DONE]\n')
    assert answer['first_token_s'] is None
    assert stream['stream']['chunks'] == []


def test_session_requests_usage_and_both_coordinators_stage_parser():
    tree = ast.parse((ROOT / 'scripts/mcdma_target/spark_glm_session.py').read_text())
    body = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == 'body' for target in node.targets))
    options = next(value for key, value in zip(body.keys, body.values) if ast.literal_eval(key) == 'stream_options')
    assert ast.literal_eval(options) == {'include_usage': True}
    for script in ('mcdma_loop_test.py', 'reverse_mcdma_test.py'):
        tree = ast.parse((ROOT / 'scripts/live' / script).read_text())
        copies = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and node.args
                  and isinstance(node.args[0], ast.Constant) and node.args[0].value == 'scp']
        assert any(any(isinstance(arg, ast.Constant) and arg.value == 'scripts/mcdma_target/glm_completion_sse.py'
                       for arg in call.args) for call in copies)


def test_continuous_summary_uses_reported_usage_not_cap(tmp_path):
    _, answer = replay(tmp_path, transcript())
    tree = ast.parse((ROOT / 'scripts/live/mcdma_loop_test.py').read_text())
    rate = next(node for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == 'glm_rate' for target in node.targets))
    namespace = {'answer': answer, 'chunks': [], 'med': lambda values: None, 'args': SimpleNamespace(glm_max_new=8)}
    exec(compile(ast.Module(body=[rate], type_ignores=[]), '<actual-coordinator-rate>', 'exec'), namespace)
    assert namespace['glm_rate']['generated_tokens'] == 3
    assert namespace['glm_rate']['max_new_tokens_cap'] == 8
    summary = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'summary' for target in node.targets))
    glm = next(value for key, value in zip(summary.keys, summary.values) if ast.literal_eval(key) == 'glm')
    fields = {ast.literal_eval(key): value for key, value in zip(glm.keys, glm.values)}
    for name in ('finish_reason', 'usage', 'stream_done'):
        assert eval(compile(ast.Expression(body=fields[name]), '<actual-coordinator-field>', 'eval'), {'answer': answer}) == answer[name]
