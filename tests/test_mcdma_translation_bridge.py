"""The native mailbox collector must feed the pinned forward translation path."""
import json
import time
from pathlib import Path

import numpy as np
import pytest

from drift.serving.bridge_files import GLM_LAYOUT
from drift.serving.bridge_recipe import PinnedRecipe
from drift.serving.mcdma_turn_outbox import McdmaTurnOutbox
from drift.serving.translation_bridge import translate_prefix
from tests.test_mcdma_turn_outbox import setup
from test_translation_bridge import digest, forward_reader


def collected(tmp_path):
    previous, bridge, native, proof = setup(tmp_path)
    config = dict(memory_root=str(previous.root), source_worker='glm', target_worker='qwen',
                  recipe_sha256='b'*64, max_raw_rows=32, max_raw_bytes=1048576,
                  max_rows=8, max_bytes=1048576, max_publications=4)
    collector = McdmaTurnOutbox(config, GLM_LAYOUT, previous.reader, previous.publisher,
                               pause=lambda _: bridge.publish_taps())
    collector.begin('glm-turn', proof, 8)
    folder = native / 'glm-turn'
    folder.mkdir(mode=0o700)
    np.savez(folder / '000000.npz', start=np.array(5), stop=np.array(14),
             **{key: np.ones((9, 512), np.float16) for key in GLM_LAYOUT})
    (folder / 'finished').write_text(json.dumps(dict(failed='', writes_scheduled=1, tap_count=1,
                                                   source_start=5, source_stop=14)))
    receipt = collector.finish('glm-turn', 8, time.monotonic()+2)
    assert bridge.forward.acknowledged()
    return Path(receipt['manifest_path'])


def translate(tmp_path, manifest, monkeypatch):
    output = tmp_path / 'translated'
    output.mkdir(mode=0o700)
    return translate_prefix(PinnedRecipe('forward', forward_reader(monkeypatch), 'b'*64),
                            manifest, digest(manifest), publication_seq=0, session='glm-turn',
                            source_worker='glm', target_worker='qwen', output=output/'forward.npz',
                            max_source_rows=8, max_source_bytes=1048576,
                            max_output_rows=16, max_output_bytes=1048576, remaining=16)


def test_real_mcdma_collector_connects_to_pinned_translator(tmp_path, monkeypatch):
    manifest = collected(tmp_path)
    result = translate(tmp_path, manifest, monkeypatch)
    assert result['source_rows'] == 2 and result['rows'] == 4
    assert result['source_manifest_sha256'] == digest(manifest)
    assert result['transport'] == 'mcdma' and result['cache_applied'] is False
    assert result['final_tail'] == 'UNKNOWN' and result['full_completion'] is False


@pytest.mark.parametrize('fault', ['transport', 'complete', 'applied', 'gap', 'bytes', 'extra',
                                 'raw_digest', 'terminal_count', 'terminal_stop', 'false_tail'])
def test_invalid_mailbox_provenance_never_reaches_translator(tmp_path, monkeypatch, fault):
    manifest = collected(tmp_path)
    report = json.loads(manifest.read_bytes())
    if fault == 'transport': report['transport'] = 'ssh'
    if fault == 'complete': report['forward_complete'] = False
    if fault == 'applied': report['cache_applied'] = True
    if fault == 'gap': report['raw_publications'][0]['start'] += 1
    if fault == 'bytes': report['raw_bytes'] += 1
    if fault == 'extra': report['text'] = 'not an activation field'
    if fault == 'raw_digest': report['raw_publications'][0]['sha256'] = None
    if fault == 'terminal_count': report['native_terminal']['tap_count'] += 1
    if fault == 'terminal_stop': report['native_terminal']['source_stop'] += 1
    if fault == 'false_tail': report['final_tail'] = 'VERIFIED_ACCEPTED_CACHE'
    manifest.chmod(0o600)
    manifest.write_text(json.dumps(report))
    with pytest.raises(ValueError): translate(tmp_path, manifest, monkeypatch)
    assert not (tmp_path/'translated/forward.npz').exists()
