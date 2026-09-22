import argparse
import json
import subprocess
import sys

import pytest

from drift.cli import dispatch, parser
from drift.transcript.importer import from_chat_messages, strict_json
from drift.transcript.process import bounded
from drift.transcript.schema import TranscriptError


def messages():
    return [{'role': 'user', 'content': 'Check the delivery time.'},
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'c1', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}]},
            {'role': 'tool', 'content': 'Arrives at 10:45.', 'tool_call_id': 'c1'},
            {'role': 'assistant', 'content': '10:45.'}]


def test_cli_import_and_validate_roundtrip(tmp_path):
    tmp_path.chmod(0o700)
    original, imported = tmp_path / 'chat.json', tmp_path / 'transcript.json'
    original.write_text(json.dumps(messages()))
    args = parser().parse_args(['transcript', 'import', '--input', str(original), '--output', str(imported),
                               '--provider', 'example', '--model', 'api-model', '--conversation-id', 'demo'])
    report = dispatch(args)
    verified = dispatch(parser().parse_args(['transcript', 'validate', '--input', str(imported)]))
    assert report['transcript_sha256'] == verified['transcript_sha256']
    assert verified['message_count'] == 4 and '10:45' not in json.dumps(verified)
    assert imported.stat().st_mode & 0o777 == 0o400
    with pytest.raises(FileExistsError):
        dispatch(args)


def test_import_rejects_unrecognized_fields_and_multimodal_content():
    for data in ([{'role': 'user', 'content': ['image']}],
                 [{'role': 'assistant', 'content': 'text', 'reasoning_content': 'private'}]):
        with pytest.raises(TranscriptError):
            from_chat_messages(data, provider='example', model='api-model', conversation_id='demo')


def test_duplicate_keys_rejected():
    with pytest.raises(TranscriptError):
        strict_json('{"content":"one","content":"two"}')


@pytest.mark.parametrize('failure', ['control', 'connect', 'timeout'])
def test_receive_transport_failure_is_structured_and_private(tmp_path, monkeypatch, failure):
    from drift.serving.handoffd_client import HandoffdClient, HandoffdError
    tmp_path.chmod(0o700)
    def unavailable(*args, **kwargs):
        if failure == 'timeout':
            raise TimeoutError('PRIVATE_TRANSPORT_SENTINEL')
        raise HandoffdError('PRIVATE_TRANSPORT_SENTINEL ' + failure)
    monkeypatch.setattr(HandoffdClient, 'pull', unavailable)
    output = tmp_path / 'received.memory'
    args = parser().parse_args(['transcript', 'receive', '--peer', 'fixture',
                               '--remote', '/memory/source.memory', '--sha256', '0' * 64,
                               '--size', '1024', '--output', str(output), '--timeout', '1'])
    report = dispatch(args)
    assert report == {'status': 'FAILED', 'error': 'TRANSCRIPT_TRANSPORT_UNAVAILABLE',
                      'cache_applied': False, 'native_recall': 'BLOCKED'}
    assert not output.exists() and not output.with_suffix('.memory.receipt.json').exists()


def test_process_timeout_reports_unknown_server_cleanup(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs['timeout'])
    monkeypatch.setattr(subprocess, 'run', timeout)
    report = bounded(argparse.Namespace(timeout=1, command='transcript', transcript_command='capture'))
    assert report == {'status': 'FAILED', 'error': 'TRANSCRIPT_WORKER_TIMEOUT',
                      'server_request_cleanup': 'UNVERIFIED'}


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf'), 601, True])
def test_invalid_time_budget_never_spawns(timeout, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('started worker')
    monkeypatch.setattr(subprocess, 'run', forbidden)
    with pytest.raises(ValueError):
        bounded(argparse.Namespace(timeout=timeout))


def test_cli_help_and_native_child_errors_do_not_echo_input(tmp_path):
    help_result = subprocess.run([sys.executable, '-m', 'drift.cli', 'transcript', '--help'],
                                 capture_output=True, text=True, timeout=10)
    assert help_result.returncode == 0
    assert all(command in help_result.stdout for command in ('import', 'validate', 'capture', 'receive', 'answer'))
    sentinel = 'PRIVATE_TRANSCRIPT_SENTINEL'
    result = subprocess.run([sys.executable, '-m', 'drift.transcript.process'],
                            input=json.dumps({'command': 'transcript', 'transcript_command': 'capture',
                                              'input': str(tmp_path / sentinel)}),
                            capture_output=True, text=True, timeout=10)
    assert sentinel not in result.stdout + result.stderr
    assert json.loads(result.stdout)['status'] == 'FAILED'
