"""Borrowed memory must be committed before any following own prompt is computed."""
from types import SimpleNamespace as NS

import pytest

from tests.test_glm_prefix_reuse import connector, live, step
from tests.test_glm_restore import Prefix, Publication, body


def test_scheduler_cap_preserves_unlinked_and_splits_only_before_boundary():
    from drift.serving.glm_prefill_boundary import cap_prefill
    request = NS(kv_transfer_params=None)
    assert cap_prefill(request, 512, 0, 64) == 512
    request.kv_transfer_params = dict(drift_session='s', drift_reserve=56,
                                     drift_reserve_start=2, drift_prefill_boundary=128)
    assert cap_prefill(request, 512, 0, 64) == 128
    assert cap_prefill(request, 512, 64, 64) == 64
    assert cap_prefill(request, 32, 0, 64) == 32
    assert cap_prefill(request, 512, 128, 64) == 512
    request.kv_transfer_params.pop('drift_session')
    request.kv_transfer_params['drift_no_link'] = True
    assert cap_prefill(request, 512, 0, 64) == 128


@pytest.mark.parametrize('boundary', [True, -1, 57, 129, 8192])
def test_invalid_boundary_never_clips_or_dispatches(boundary):
    from drift.serving.glm_prefill_boundary import cap_prefill
    request = NS(kv_transfer_params=dict(drift_session='s', drift_reserve=56,
                                        drift_reserve_start=2, drift_prefill_boundary=boundary))
    with pytest.raises(ValueError): cap_prefill(request, 512, 0, 64)


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
