"""Acquire one controlled Qwen fixture under the shared on-host supervisor, without printing data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from drift.serving.fixture_host import claim_fixture,host_snapshot
from drift.serving.fixture_supervision import run_fixture_child
from drift.serving.fixture_receipt import validate_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('private-root','checkpoint','translator','tap-script'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    began=time.monotonic(); deadline=began+300
    claim=claim_fixture(args.private_root)
    result={'status':'INVALID','fixture_id':claim.name,'scope':'controlled_qwen_source_fixture',
            'max_wall_seconds':300,'max_own_prompt_tokens':32,'generated_tokens':0,'model_requests_cap':1,
            'allocator_memory':'NOT_MEASURED','before':host_snapshot(),'supervision':None}
    try:
        before=result['before']
        if before['lock_available'] is not True or before['reclaimable_gib'] is None or before['reclaimable_gib']<188:
            result.update(status='BLOCKED',reason='STUDIO_ADMISSION_UNAVAILABLE')
        else:
            command=[sys.executable,str(Path(__file__).with_name('fixture_child.py').resolve()),
                     '--out',str(claim/'raw'),'--checkpoint',str(args.checkpoint.expanduser().resolve()),
                     '--translator',str(args.translator.expanduser().resolve()),'--tap-script',str(args.tap_script.resolve())]
            supervisor=run_fixture_child(command,deadline=deadline,work_seconds=min(280,deadline-time.monotonic()-10))
            result['supervision']=supervisor
            valid=(supervisor['reason']=='child_exit' and supervisor['returncode']==0
                   and supervisor['child_reaped'] is True and supervisor['process_group_alive'] is False
                   and supervisor['supervisor_returncode']==0)
            if not valid: raise RuntimeError('fixture child did not complete safely')
            path=claim/'raw/fixture.json'
            if path.stat().st_size>8192: raise ValueError('fixture receipt exceeds bound')
            receipt=validate_receipt(json.loads(path.read_text()))
            output=claim/'raw/publication.npz'
            if output.stat().st_size>1048576 or hashlib.sha256(output.read_bytes()).hexdigest()!=receipt['fixture_sha256']:
                raise ValueError('fixture output differs from receipt')
            result.update(status='PASSED',fixture=receipt)
    except Exception as error:
        result.update(status='INVALID',reason='FIXTURE_ACQUISITION_FAILED',error_type=type(error).__name__)
    result['after']=host_snapshot(timeout=min(2,max(.01,deadline-time.monotonic())))
    result['total_seconds']=time.monotonic()-began
    if result['status']=='PASSED' and (result['after']['lock_available'] is not True
            or result['after']['reclaimable_gib'] is None or result['total_seconds']>=300):
        result.update(status='INVALID',reason='POST_EXIT_EVIDENCE_INCOMPLETE')
    with (claim/'receipt.json').open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)
    return 0 if result['status']=='PASSED' else 2


if __name__=='__main__':
    raise SystemExit(main())
