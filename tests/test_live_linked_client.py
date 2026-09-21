"""An explicit linked client controls one gated stream and discards generated text."""
import json
import sys

import pytest

from drift.serving.live_linked_client import LinkedSession
from drift.serving.live_qualification_client import Limits, OwnedSession, QualificationError, claim_session
from test_live_qualification_client import until


def client_for(tmp_path, source):
    script = tmp_path / 'runner.py'
    script.write_text(source)
    claim = claim_session(tmp_path / 'claims')
    limits = Limits(max_new=8, max_prompt_tokens=256, max_seconds=5, stop_timeout=.1)
    command = [sys.executable, str(script), '--session', claim.name, '--max-new', '8', '--reserve', '12',
               '--max-prompt-tokens', '256', '--wait-publication', '--ready-timeout', '4', '--no-tap', '--control-stdin']
    return LinkedSession(command, claim, limits)


def test_linked_ready_is_not_activity_and_release_is_required(tmp_path):
    source = "import sys; print('{\"publication_ready\":true}',flush=True); sys.stdin.readline(); print('{\"started\":true,\"own_prompt_tokens\":2}',flush=True); print('{\"text\":\"PRIVATE_SYNTHETIC\"}',flush=True); sys.stdin.readline(); print('{\"cancelled\":true}',flush=True); raise SystemExit(2)"
    client = client_for(tmp_path, source)
    client.start()
    try:
        until(client, lambda: client.publication_ready)
        assert not client.request_state().active
        client.release()
        until(client, lambda: client.request_state().active)
        client.abort()
        until(client, lambda: client.request_state().cancelled)
        assert client.report()['release_sent'] and 'PRIVATE_SYNTHETIC' not in json.dumps(client.report())
    finally:
        client.close()


def test_linked_command_is_still_rejected_by_no_link_client(tmp_path):
    client = client_for(tmp_path, '')
    with pytest.raises(QualificationError): OwnedSession(client.command, client.claim, client.limits)


@pytest.mark.parametrize('fault', ['missing_no_tap', 'no_link', 'early_release', 'duplicate_release', 'output_before_release'])
def test_linked_protocol_refuses_scope_or_order_changes(tmp_path, fault):
    source = "import sys,time; print('{\"publication_ready\":true}',flush=True); "
    source += "print('{\"text\":\"PRIVATE_SYNTHETIC\"}',flush=True); " if fault == 'output_before_release' else ''
    source += 'time.sleep(10)'
    client = client_for(tmp_path, source)
    if fault in {'missing_no_tap', 'no_link'}:
        command = [arg for arg in client.command if arg != '--no-tap'] if fault == 'missing_no_tap' else [*client.command, '--no-link']
        with pytest.raises(QualificationError): LinkedSession(command, client.claim, client.limits)
        return
    client.start()
    try:
        if fault == 'early_release':
            with pytest.raises(QualificationError): client.release()
        elif fault == 'output_before_release':
            with pytest.raises(QualificationError): until(client, lambda: False)
        else:
            until(client, lambda: client.publication_ready)
            client.release()
            with pytest.raises(QualificationError): client.release()
    finally:
        client.close()


def test_prompt_cap_is_checked_before_readiness_or_completion(gated_runner):
    start, requests, _ = gated_runner
    process = start('--reserve', '12', '--max-prompt-tokens', '12')
    process.wait(timeout=3)
    events = [json.loads(row) for row in process.stdout.read().splitlines()]
    assert process.returncode != 0 and requests == []
    assert not any(event.get('publication_ready') for event in events)


from test_live_publication_gate import gated_runner
