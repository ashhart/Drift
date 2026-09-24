"""Borrowed memory must be committed before any following own prompt is computed."""
from types import SimpleNamespace as NS

import pytest

from tests.test_glm_prefix_reuse import connector, live, step
from tests.test_glm_restore import Prefix, Publication, body


def aligned_at(start, block):
    """vLLM's aligned split for a chunk that is not the prompt's last: its end floors to a block boundary."""
    return lambda count: max((start+count)//block*block-start, 0)


def test_split_preserves_unlinked_and_stops_exactly_at_the_boundary():
    from drift.serving.glm_prefill_boundary import split
    request = NS(kv_transfer_params=None)
    assert split(request, 512, 0, 3584, aligned_at(0, 3584)) == 0              # unlinked: vLLM decides
    request.kv_transfer_params = dict(drift_session='s', drift_reserve=56,
                                     drift_reserve_start=2, drift_prefill_boundary=58)
    assert split(request, 512, 0, 3584, aligned_at(0, 3584)) == 58             # the reserve's end, inside the block
    assert split(request, 30, 0, 3584, aligned_at(0, 3584)) == 0               # cannot reach it: vLLM aligns
    assert split(request, 512, 58, 3584, lambda count: count) == 512           # past it: vLLM decides
    request.kv_transfer_params.pop('drift_session')
    request.kv_transfer_params['drift_no_link'] = True
    assert split(request, 512, 0, 3584, aligned_at(0, 3584)) == 58


def test_split_aligns_at_a_block_boundary_before_a_later_stop():
    from drift.serving.glm_prefill_boundary import split
    request = NS(kv_transfer_params=dict(drift_session='s', drift_reserve=3000,
                                        drift_reserve_start=1000, drift_prefill_boundary=5000))
    assert split(request, 7168, 0, 3584, aligned_at(0, 3584)) == 3584          # the block's state is materialized first
    assert split(request, 7168, 3584, 3584, aligned_at(3584, 3584)) == 1416


@pytest.mark.parametrize('boundary', [True, -1, 57, 8192])
def test_invalid_boundary_never_clips_or_dispatches(boundary):
    from drift.serving.glm_prefill_boundary import split
    request = NS(kv_transfer_params=dict(drift_session='s', drift_reserve=56,
                                        drift_reserve_start=2, drift_prefill_boundary=boundary))
    with pytest.raises(ValueError): split(request, 512, 0, 3584, aligned_at(0, 3584))


def test_a_state_needs_the_boundary_at_the_reserve_end(connector):
    owner, _ = connector
    owner.on_new_request(live('a', start=2, reserve=56, prompt=300, name='x', drift_tap=False,
                              drift_prefill_boundary=64, drift_state_blob='own-1'))
    assert "reserve's end" in owner._live['a']['failed']
    owner.on_new_request(live('b', start=2, reserve=56, prompt=300, name='y', drift_tap=False,
                              drift_prefill_boundary=58, drift_state_blob='../x'))
    assert 'must match' in owner._live['b']['failed']
    owner.on_new_request(live('c', start=2, reserve=56, prompt=300, name='z', drift_tap=False,
                              drift_prefill_boundary=58, drift_state_blob='own-1'))
    assert owner._live['c']['failed'] == '' and owner._live['c']['state_blob'] == 'own-1'


@pytest.mark.parametrize('scheduled,publication', [(129, True), (128, False)])
def test_connector_refuses_late_or_absent_initial_memory(connector, scheduled, publication):
    owner, root = connector
    owner.on_new_request(live('r', start=2, reserve=56, prompt=300,
                              drift_tap=False, drift_prefill_boundary=128))
    folder = root/'tp-live-in'/'s'; folder.mkdir(parents=True)
    if publication: (folder/'000000.npz').touch()
    metadata = owner.build_connector_meta(step(new=[('r', ([0, 1, 2],), 0)], scheduled={'r': scheduled}))
    assert metadata.live[0].failed and not metadata.live[0].apply


def test_restoration_keeps_native_chat_and_counts_padding(tmp_path):
    from drift.serving.glm_restore_turn import TurnRestorer
    restorer = TurnRestorer(Prefix(), lambda _: Publication([]), rows=56, digest='a'*64,
                           root=tmp_path, deadline=lambda: 1e20, max_input_bytes=65536,
                           causal_prefill=True)
    original = body(); extra = restorer.prepare(original)
    proof = restorer.plan['proof']
    assert extra['kv_transfer_params']['drift_prefill_boundary'] == 128
    assert proof['reserve_tokens'] == 56 and proof['padding_tokens'] == 69
    assert proof['prompt_tokens'] == proof['own_tokens']+56+69
    request = restorer.request({**original, **extra})
    assert request['messages'][0]['content'] == '[MASK]'*125+original['messages'][0]['content']
    assert request['messages'][1:] == original['messages'][1:] and request['tools'] == original['tools']


def test_unlinked_control_uses_same_padding_and_scheduler_cap(tmp_path):
    from drift.serving.glm_restore_turn import TurnRestorer
    calls = []
    restorer = TurnRestorer(Prefix(), calls.append, rows=56, digest='a'*64,
                           root=tmp_path, deadline=lambda: 1e20, max_input_bytes=65536,
                           causal_prefill=True, linked=False)
    extra = restorer.prepare(body())
    assert extra['kv_transfer_params'] == dict(drift_no_link=True, drift_reserve=56,
                                              drift_reserve_start=3, drift_prefill_boundary=128)
    assert not calls
