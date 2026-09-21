"""Keep native GLM tap evidence bounded and distinct from a complete final cache."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pytest


def setup(tmp_path):
    from drift.serving.glm_turn_outbox import TurnOutbox
    output = tmp_path/'exports'; output.mkdir(mode=0o700)
    native = tmp_path/'native'; native.mkdir(mode=0o700)
    config = dict(memory_root=str(output), source_worker='GLM', target_worker='Qwen', recipe_sha256='a'*64,
                  max_raw_rows=32, max_raw_bytes=65536, max_rows=8, max_bytes=32768, max_publications=4)
    collector = TurnOutbox(config, {'l3': (512,)}, native)
    folder = native/'turn-1'; folder.mkdir(mode=0o700)
    proof = dict(verified=True, own_tokens=10, prompt_tokens=12, reserve_start=3, reserve_tokens=2)
    return collector, folder, proof


def tap(folder, seq, start, stop):
    path = folder/f'{seq:06d}.npz'
    np.savez(path, start=np.array(start), stop=np.array(stop), l3=np.full((stop-start,512),seq+1,dtype=np.float16))
    return path


def finished(folder):
    (folder/'finished').write_text(json.dumps({'failed':'','writes_scheduled':1}))


def test_excludes_foreign_span_and_reconstructed_prefill_and_declares_missing_tail(tmp_path):
    collector, folder, proof = setup(tmp_path)
    collector.begin('turn-1', proof, 8)
    tap(folder,0,5,12); tap(folder,1,12,17); finished(folder)
    report = collector.finish('turn-1',8,time.monotonic()+2)
    assert report['evidence']=='EXPORTED_PREFIX' and report['full_completion'] is False
    assert report['final_tail']=='UNKNOWN' and report['selected_rows']==5
    assert report['verified_stop']==17 and report['prompt_tokens']==12
    manifest=Path(report['manifest_path'])
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==report['manifest_sha256']
    details=json.loads(manifest.read_text())
    assert details['publications'][0]['source_start']==12
    assert details['publications'][0]['source_stop']==17
    selected=manifest.parent/details['publications'][0]['file']
    with np.load(selected) as data:
        assert set(data.files)=={'l3'} and data['l3'].shape==(5,512)
    assert len(details['raw_publications'])==2


def test_finished_marker_does_not_imply_any_generated_row_was_exported(tmp_path):
    collector,folder,proof=setup(tmp_path);collector.begin('turn-1',proof,8)
    tap(folder,0,5,12);finished(folder)
    report=collector.finish('turn-1',8,time.monotonic()+2)
    assert report['selected_rows']==0 and report['full_completion'] is False
    assert report['evidence']=='EMPTY_EXPORTED_PREFIX'


@pytest.mark.parametrize('fault',['gap','foreign','bound','error','tail','reuse','hash'])
def test_invalid_outbox_never_produces_a_success_manifest(tmp_path,fault):
    collector,folder,proof=setup(tmp_path);collector.begin('turn-1',proof,8)
    path=tap(folder,0,5,17);finished(folder)
    if fault=='gap':path.rename(folder/'000001.npz')
    if fault=='foreign':tap(folder,0,4,17)
    if fault=='bound':tap(folder,0,5,21)
    if fault=='error':(folder/'error.rank1').write_text('fixed error')
    if fault=='tail':(folder/'.000001.tmp.npz').write_bytes(b'incomplete')
    if fault=='reuse':
        with pytest.raises(ValueError):collector.begin('turn-1',proof,8)
        return
    if fault=='hash':path.write_bytes(b'corrupted')
    with pytest.raises(ValueError):collector.finish('turn-1',8,time.monotonic()+2)


def test_missing_finish_marker_obeys_absolute_deadline(tmp_path):
    collector,folder,proof=setup(tmp_path);collector.begin('turn-1',proof,8)
    with pytest.raises(TimeoutError):collector.finish('turn-1',8,time.monotonic()+.03)


def test_manifest_does_not_represent_new_own_tool_results_as_generated_memory(tmp_path):
    collector,folder,proof=setup(tmp_path);proof.update(own_tokens=18,prompt_tokens=20)
    collector.begin('turn-1',proof,8)
    tap(folder,0,5,22);finished(folder)
    report=collector.finish('turn-1',8,time.monotonic()+2)
    assert report['selected_rows']==2 and report['new_own_inputs']=='NOT_EXPORTED'
    assert report['scope']=='newly_generated_only'


def test_foreign_session_and_overflowing_half_precision_are_rejected(tmp_path):
    collector,folder,proof=setup(tmp_path);collector.begin('turn-1',proof,8)
    with pytest.raises(ValueError):collector.finish('turn-2',8,time.monotonic()+2)
    other=tmp_path/'other';other.mkdir()
    collector,folder,proof=setup(other);collector.begin('turn-1',proof,8)
    np.savez(folder/'000000.npz',start=np.array(5),stop=np.array(17),l3=np.full((12,512),1e10,dtype=np.float32))
    finished(folder)
    with pytest.raises(ValueError):collector.finish('turn-1',8,time.monotonic()+2)
