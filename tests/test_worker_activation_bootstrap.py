"""Exercise receipt-bound fresh-child wake through the actual worker dispatch."""
import hashlib

import numpy as np
import pytest

from test_worker_activation_wake import SyntheticBoundary


def prepared(tmp_path, choice=4):
    fixture = SyntheticBoundary(tmp_path, allow_activation_wake=True)
    path = fixture.activation.root / 'incoming.npz'
    np.savez(path, k0=np.zeros((2, 1, 2), np.float16), v0=np.full((2, 1, 2), choice, np.float16))
    fixture.publication['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = fixture.publish()
    return fixture, {'activation_seq': receipt['seq'], 'foreign_total': receipt['foreign_total']}


@pytest.mark.parametrize('choice', [4, 5])
def test_fresh_wake_first_token_depends_on_kv_without_peer_prompt(tmp_path, choice):
    fixture, binding = prepared(tmp_path, choice)
    ack = fixture.events(1, 'activation_wake', binding)
    assert ack[0]['op'] == 'activation_wake_ack'
    assert ack[0]['payload'] == binding and fixture.calls == []
    assert fixture.session.ready and not fixture.session.control_ready
    events = fixture.events(2, 'stream', {'max_tokens': 1})
    assert events[0]['payload']['text'] == str(choice)
    assert fixture.calls == [([1, 2], 2), ([choice], 2)]
    assert fixture.state['own_slots'] == [2, 3, 4]
    assert fixture.state['cache'][0].offset == 5
    assert fixture.backend.codec.consumed == [1, 2, choice]
    assert events[-1]['payload']['usage'] == {'input_tokens': 2, 'output_tokens': 1}
    assert fixture.session.tokens == 3 and fixture.session.turns == 1
    assert not any(message['role'] == 'user' for message in fixture.backend.codec.messages)
    assert 'extend' not in fixture.operations


@pytest.mark.parametrize('mutation', [
    {'activation_seq': 2}, {'foreign_total': 3}, {'activation_seq': True},
    {'text': 'forbidden peer payload'}, {'foreign_total': 0},
])
def test_wake_rejects_wrong_receipt_or_text_before_consuming_tokens(tmp_path, mutation):
    fixture, binding = prepared(tmp_path)
    event = fixture.events(1, 'activation_wake', {**binding, **mutation})[0]
    assert event['op'] == 'error' and fixture.session.poisoned
    assert fixture.calls == [] and fixture.session.tokens == 0


@pytest.mark.parametrize('mode', ['disabled', 'missing', 'native', 'control', 'stopped', 'tools', 'wrong_session'])
def test_wake_unsupported_states_fail_closed(tmp_path, mode):
    fixture, binding = prepared(tmp_path)
    sequence = 1
    if mode == 'disabled':
        fixture.session.allow_activation_wake = False
    elif mode == 'missing':
        fixture.state.pop('activation_receipt', None)
    elif mode == 'native':
        fixture.events(1, 'own_prompt', {'text': 'receiver local input'})
        sequence = 2
    elif mode == 'control':
        fixture.backend.pending_control = [{'role': 'system', 'content': 'receiver local update'}]
    elif mode == 'stopped':
        fixture.state.update(done=True, pending_stop=9)
    elif mode == 'tools':
        fixture.backend.awaiting_tools = 1
    else:
        fixture.state['activation_receipt'] = {'session': 'other', 'target_worker': 'qwen', **binding}
    before = len(fixture.calls)
    assert fixture.events(sequence, 'activation_wake', binding)[0]['op'] == 'error'
    assert fixture.session.poisoned and len(fixture.calls) == before


def test_new_publication_after_wake_invalidates_receipt_before_bootstrap(tmp_path):
    fixture, binding = prepared(tmp_path)
    assert fixture.events(1, 'activation_wake', binding)[0]['op'] == 'activation_wake_ack'
    fixture.publication['seq'] = 2
    fixture.publish()
    assert fixture.events(2, 'stream', {'max_tokens': 1})[0]['op'] == 'error'
    assert fixture.calls == [] and fixture.session.tokens == 0


def test_bootstrap_and_output_budget_checked_before_native_consumption(tmp_path):
    fixture, binding = prepared(tmp_path)
    fixture.backend.limits = {**fixture.backend.limits, 'max_session_tokens': 2}
    assert fixture.events(1, 'activation_wake', binding)[0]['op'] == 'activation_wake_ack'
    assert fixture.events(2, 'stream', {'max_tokens': 1})[0]['op'] == 'error'
    assert fixture.calls == [] and fixture.session.tokens == 0


def test_bootstrap_stop_is_charged_once_and_cannot_wake_again(tmp_path):
    fixture, binding = prepared(tmp_path, 9)
    assert fixture.events(1, 'activation_wake', binding)[0]['op'] == 'activation_wake_ack'
    events = fixture.events(2, 'stream', {'max_tokens': 1})
    assert events[-1]['payload']['usage'] == {'input_tokens': 2, 'output_tokens': 1}
    assert fixture.calls == [([1, 2], 2)]
    assert fixture.backend.codec.consumed == [1, 2, 9]
    assert fixture.state['pending_stop'] == 9
    assert fixture.events(3, 'activation_wake', binding)[0]['op'] == 'error'
    assert fixture.session.tokens == 3


def test_setup_template_change_is_rejected_before_consumption(tmp_path):
    fixture, binding = prepared(tmp_path)
    fixture.tokenizer.apply_chat_template = lambda *args, **kwargs: [6, 7]
    assert fixture.events(1, 'activation_wake', binding)[0]['op'] == 'activation_wake_ack'
    assert fixture.events(2, 'stream', {'max_tokens': 1})[0]['op'] == 'error'
    assert fixture.calls == []
