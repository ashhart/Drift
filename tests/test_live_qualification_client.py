"""Aggregate-only qualification uses local fakes and a loopback SSE server."""
import json
import os
from pathlib import Path
import sys
import time

import pytest

from drift.serving.live_qualification_client import Limits, OwnedSession, QualificationError, claim_session
from test_live_session_control import sse_server

ROOT = Path(__file__).resolve().parents[1]


def fake_client(tmp_path, source, **limits):
    runner = tmp_path / 'fake.py'
    runner.write_text(source)
    budget = Limits(**limits)
    claim = claim_session(tmp_path / 'claims')
    command = [sys.executable, str(runner), '--session', claim.name, '--max-new', str(budget.max_new),
               '--reserve', '8', '--no-link', '--control-stdin']
    return OwnedSession(command, claim, budget)


def until(client, predicate):
    for _ in range(200):
        client.poll(0.02)
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError('synthetic state did not arrive')


START = "print('{\"started\":true,\"own_prompt_tokens\":1,\"span_start\":0}',flush=True); "
OUTPUT = "print('{\"text\":\"PRIVATE_SYNTHETIC_CONTENT\"}',flush=True); "


def test_launch_is_acknowledged_before_delayed_tokenization(tmp_path):
    client = fake_client(tmp_path, 'import time; time.sleep(30)')
    client.start()
    try:
        assert client.request_state().started
        assert not client.request_state().active
        assert client.report()['started_events'] == 0
    finally:
        client.close()


def test_exit_two_without_typed_acknowledgement_is_not_cancellation(tmp_path):
    client = fake_client(tmp_path, 'import sys; ' + START + OUTPUT + 'sys.stdin.readline(); raise SystemExit(2)')
    client.start()
    try:
        until(client, lambda: client.request_state().active)
        client.abort()
        with pytest.raises(QualificationError, match='CHILD_EXIT_PROTOCOL'):
            until(client, lambda: False)
        assert not client.request_state().cancelled
    finally:
        client.close()



def test_real_runner_cancellation_is_acknowledged_without_retaining_text(tmp_path, sse_server, monkeypatch):
    base, disconnected, _ = sse_server
    monkeypatch.setenv('DRIFT_GLM_KEY', 'synthetic-key')
    monkeypatch.setenv('PYTHONPATH', str(ROOT))
    messages = tmp_path / 'messages.json'
    messages.write_text('[{"role":"user","content":"public synthetic input"}]')
    limits = Limits(max_new=8)
    claim = claim_session(tmp_path / 'claims')
    command = [sys.executable, str(ROOT / 'scripts/live/spark_live_session.py'), '--messages', str(messages),
               '--session', claim.name, '--max-new', '8', '--reserve', '8', '--base', base, '--no-link', '--control-stdin']
    client = OwnedSession(command, claim, limits)
    client.start()
    try:
        until(client, lambda: client.request_state().active)
        assert client.report()['started_events'] == 1
        client.abort()
        until(client, lambda: client.request_state().cancelled)
        assert disconnected.wait(2)
        report = client.report()
        assert report['child_returncode'] == 2 and report['child_terminated'] and report['stdout_closed']
        assert report['requested_max_new'] == 8 and report['actual_generated_tokens'] is None
        assert report['output_chunks'] >= 1 and report['output_bytes'] > 0
        assert 'synthetic chunk' not in json.dumps(report)
        with pytest.raises(ProcessLookupError):
            os.kill(report['child_pid'], 0)
    finally:
        client.close()


@pytest.mark.parametrize('terminal', ["print('{\"done\":true}',flush=True)",
                                    "print('{\"cancelled\":true}',flush=True); raise SystemExit(2)"])
def test_natural_or_unsolicited_completion_cannot_qualify_as_owned_abort(tmp_path, terminal):
    client = fake_client(tmp_path, START + OUTPUT + terminal)
    client.start()
    try:
        try:
            until(client, lambda: client.report()['child_terminated'])
        except QualificationError:
            pass
        assert not client.request_state().cancelled
        with pytest.raises(QualificationError):
            client.abort()
        assert not client.abort_sent
    finally:
        client.close()


@pytest.mark.parametrize('source,limits,code', [
    ('import time; ' + START + 'time.sleep(30)', {'max_seconds': 0.08}, 'WALL_LIMIT'),
    (START + "print('x'*100,flush=True)", {'max_line_bytes': 64}, 'LINE_LIMIT'),
    (START + OUTPUT, {'max_output_bytes': 1}, 'OUTPUT_BYTE_LIMIT'),
    (START + OUTPUT, {'max_events': 1}, 'EVENT_LIMIT'),
    (START, {'max_prompt_tokens': 0}, 'LIMIT_VALIDATION'),
])
def test_limits_fail_closed_without_output_in_errors(tmp_path, source, limits, code):
    if code == 'LIMIT_VALIDATION':
        with pytest.raises(ValueError):
            fake_client(tmp_path, source, **limits)
        return
    client = fake_client(tmp_path, source.rstrip('; ') + '; import time; time.sleep(30)', **limits)
    client.start()
    try:
        with pytest.raises(QualificationError, match=code):
            until(client, lambda: False)
        assert client.report()['child_terminated']
        assert 'PRIVATE_SYNTHETIC_CONTENT' not in json.dumps(client.report())
    finally:
        client.close()


def test_started_and_empty_output_are_insufficient_activity(tmp_path):
    client = fake_client(tmp_path, START + "print('{\"text\":\"\"}',flush=True); import time; time.sleep(30)")
    client.start()
    try:
        until(client, lambda: client.report()['output_chunks'] == 1)
        assert client.request_state().started and not client.request_state().active
        with pytest.raises(QualificationError, match='NOT_ACTIVE'):
            client.abort()
        assert not client.abort_sent
    finally:
        client.close()


def test_session_claims_are_unique_and_cannot_be_restarted(tmp_path):
    first, second = claim_session(tmp_path / 'claims'), claim_session(tmp_path / 'claims')
    assert first.name != second.name
    first.spend()
    with pytest.raises(QualificationError, match='SESSION_REUSE'):
        first.spend()


def test_command_cannot_enable_activation_or_increase_requested_tokens(tmp_path):
    claim = claim_session(tmp_path / 'claims')
    for flags in ([], ['--no-link'], ['--no-link', '--control-stdin', '--max-new', '1000']):
        with pytest.raises(QualificationError, match='COMMAND_CONTRACT'):
            OwnedSession(['runner', '--session', claim.name, '--max-new', '8', *flags], claim, Limits(max_new=8))
