"""Admit accepted-cache terminal evidence without claiming every output token cached."""
import json
import time
from pathlib import Path

import pytest

from test_glm_turn_outbox import setup, tap
from test_vllm_glm53_inject import connector


def terminal(folder, **changes):
    marker = dict(failed='', writes_scheduled=1, tap_count=2, source_start=5, source_stop=17)
    (folder / 'finished').write_text(json.dumps(marker | changes))


def test_verified_terminal_is_admitted_and_retained_in_manifest(tmp_path):
    collector, folder, proof = setup(tmp_path)
    collector.begin('turn-1', proof, 8)
    tap(folder, 0, 5, 12)
    tap(folder, 1, 12, 17)
    terminal(folder)
    report = collector.finish('turn-1', 8, time.monotonic() + 2)
    assert report['final_tail'] == 'VERIFIED_ACCEPTED_CACHE'
    assert report['full_completion'] is False
    assert report['selected_rows'] == 5
    manifest = json.loads(Path(report['manifest_path']).read_bytes())
    assert manifest['final_tail'] == 'VERIFIED_ACCEPTED_CACHE'


@pytest.mark.parametrize('changes', [dict(tap_count=1), dict(tap_count=True),
    dict(source_start=4), dict(source_stop=18), dict(source_stop=16),
    dict(source_stop=True), dict(extra=1), dict(failed='failed')])
def test_terminal_must_match_actual_contiguous_export(tmp_path, changes):
    collector, folder, proof = setup(tmp_path)
    collector.begin('turn-1', proof, 8)
    tap(folder, 0, 5, 12)
    tap(folder, 1, 12, 17)
    terminal(folder, **changes)
    with pytest.raises(ValueError, match='NATIVE_OUTBOX_FAILED'):
        collector.finish('turn-1', 8, time.monotonic() + 2)
    assert not (collector.root / 'turn-1' / 'manifest.json').exists()


def test_partial_terminal_schema_cannot_fall_back_to_legacy(tmp_path):
    collector, folder, proof = setup(tmp_path)
    collector.begin('turn-1', proof, 8)
    tap(folder, 0, 5, 17)
    (folder / 'finished').write_text(json.dumps(dict(failed='', writes_scheduled=1, source_stop=17)))
    with pytest.raises(ValueError, match='NATIVE_OUTBOX_FAILED'):
        collector.finish('turn-1', 8, time.monotonic() + 2)


def test_native_connector_tail_reaches_new_owner_outbox(connector):
    from drift.serving.glm_turn_outbox import TurnOutbox
    from test_live_tap_tail import finished_request, run_steps
    c, root = connector
    output = root / 'owner-exports'
    output.mkdir(mode=0o700)
    folder = root / 'tp-live-out' / 'tail-test'
    folder.mkdir(parents=True, mode=0o700)
    config = dict(memory_root=str(output), source_worker='GLM', target_worker='Qwen',
                  recipe_sha256='a' * 64, max_raw_rows=32, max_raw_bytes=65536,
                  max_rows=8, max_bytes=32768, max_publications=4)
    collector = TurnOutbox(config, {'l3': (512,), 'l7': (512,)}, c._live_out)
    collector.begin('tail-test', dict(verified=True, prompt_tokens=10,
                    reserve_start=0, reserve_tokens=0), 8)
    run_steps(c)
    c.request_finished(finished_request(), ())
    report = collector.finish('tail-test', 8, time.monotonic() + 2)
    assert report['final_tail'] == 'VERIFIED_ACCEPTED_CACHE'
    assert report['selected_rows'] == 2 and report['verified_stop'] == 12
    assert report['native_terminal'] == dict(tap_count=2, source_start=0, source_stop=12)
