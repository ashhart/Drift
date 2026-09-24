"""Exercise real installed-OMP task delegation without models or native transport."""
import argparse
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
from kv_subagent_setup import stage, save
from probe import probe_environment, sandbox_policy
from provider_probe_process import ProbeProcess
from registration_setup import digest


def run(omp, source, production_command=False):
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='drift-kv-') as directory:
        root = Path(directory).resolve(); root.chmod(0o700)
        task, control, owner, boundary, exchange = stage(root, source, production_command=production_command)
        listener = socket.socket(socket.AF_UNIX); listener.bind(exchange['socket'])
        os.chmod(exchange['socket'],0o600); listener.listen(2); listener.settimeout(.05)
        stop = threading.Event(); calls = []
        def serve():
            while not stop.is_set():
                try: peer,_ = listener.accept()
                except socket.timeout: continue
                with peer:
                    peer.settimeout(1); raw = b''
                    while b'\n' not in raw and len(raw)<4096:
                        block = peer.recv(4096)
                        if not block: break
                        raw += block
                    if not raw: continue
                    frame = json.loads(raw); entry = next(item for item in exchange['workers'] if item['route']==frame['route'])
                    if frame['op'] == 'status':
                        peer.sendall(json.dumps(dict(op='status',route=frame['route'],mode='drift',sequence=0,foreign_rows=0,poisoned=False)).encode()+b'\n')
                        continue
                    if frame['op'] == 'confirm':
                        record = pending[frame['route']]
                        peer.sendall(json.dumps(dict(op='confirmed',route=frame['route'],published=record)).encode()+b'\n')
                        continue
                    sequence = calls.count(frame['route']); calls.append(frame['route'])
                    record = dict(v=1,session=entry['exchange_session'],sequence=sequence,mode='drift',
                                  source_worker=entry['source_worker'],target_worker=entry['target_worker'],rows=1,source_rows=1,
                                  copies=1,bytes=64,source_sha256='b'*64,text_bytes=0,applied_ranks=entry['ranks'],receipts=['c'*64])
                    if frame['op'] == 'stage':
                        pending[frame['route']] = record
                        staged = {key:value for key,value in record.items() if key not in ('applied_ranks','receipts')}
                        reply = dict(op='staged',route=frame['route'],delivery=staged | dict(state='STAGED_NOT_APPLIED'))
                    else: reply = dict(op='exchanged',route=frame['route'],published=record)
                    peer.sendall(json.dumps(reply).encode()+b'\n')
        pending = {}
        thread = threading.Thread(target=serve); thread.start()
        env = probe_environment(task) | dict(DRIFT_WORKER_CONFIG=str(owner),DRIFT_WORKER_CONFIG_SHA256=digest(owner),
            DRIFT_PROVIDER_BOUNDARY_CONFIG=str(control/'boundary.json'),DRIFT_PROVIDER_BOUNDARY_SHA256=digest(control/'boundary.json'),
            DRIFT_EXCHANGE_CONFIG=str(control/'exchange.json'),DRIFT_EXCHANGE_SHA256=digest(control/'exchange.json'))
        command = [str(omp.resolve()),'--cwd',str(task),'--no-session','--tools','task,yield','--no-lsp','--no-pty',
                   '--no-extensions','--no-skills','--no-rules','--no-title','--no-prewalk','--extension',
                   str(task/('plugin/omp-drift/src/omp.ts' if production_command else 'scripts/omp/experimental_extension.mjs')),'--model','drift-experimental/glm-fixture',
                   '--thinking','off','--max-time','10','--print','--mode','json','Own deterministic fixture task.']
        if production_command:
            command = command[:-6] + ['--max-time','10','--mode','rpc']
        policy = root/'policy.sb'
        policy.write_text(sandbox_policy(root, Path.home()) + '\n(allow network* (local unix-socket))\n'
                          + '(allow file-read-data (subpath '+json.dumps(str(Path(os.sys.executable).resolve().parent.parent))+'))\n')
        actor = ProbeProcess(['/usr/bin/sandbox-exec','-f',str(policy),*command], task, env, rpc=production_command); released = 0
        if production_command: actor.send(dict(type='prompt',id='enable',message='/drift subagent enable'))
        prompted = False; ended_at = None
        try:
            while actor.process.poll() is None and time.monotonic()-started<12:
                if production_command:
                    if actor.responses.get('enable') and not prompted:
                        actor.send(dict(type='prompt',id='task',message='Own deterministic fixture task.')); prompted=True
                    released = sum((control/f'parent-{epoch}-release.json').exists() for epoch in range(2))
                    if actor.agent_end:
                        if ended_at is None: ended_at = time.monotonic()
                        if time.monotonic()-ended_at > .5: actor.eof()
                    time.sleep(.01); continue
                paths = [control/f'{role}-{released}-{kind}.json' for role in ('parent','child') for kind in ('ready','exchange')]
                if released<2 and all(path.exists() for path in paths):
                    for index,role in enumerate(('parent','child')):
                        save(control/f'{role}-{released}-release.json',dict(v=2,epoch=released,action='resume',nonce=boundary['nonce'],
                             ready_sha256=digest(paths[index*2]),peer_ready_sha256=digest(paths[(1-index)*2]),exchange_sha256=digest(paths[index*2+1])))
                    released+=1
                time.sleep(.01)
        finally:
            cleanup=actor.close(time.monotonic()+3);stop.set();thread.join(timeout=2);listener.close()
        facts = {name:json.loads(path.read_bytes()) if (path:=task/(name+'-worker.json')).exists() else {} for name in ('glm-fixture','qwen-fixture')}
        counts = {'glm-fixture':dict(open=1,own_prompt=1,stream=4,tool_result=3,close=1),
                  'qwen-fixture':dict(open=1,own_prompt=1,stream=3,tool_result=2,close=1)}
        passed = actor.process.returncode==0 and released==2 and not actor.error_codes and facts == counts
        passed = passed and all((control/(role+'-settled.json')).exists() for role in ('parent','child'))
        passed = passed and all(cleanup.get(key) for key in ('reaped','joined','group_gone')) and not cleanup['terminated']
        return dict(verdict='PASSED' if passed else 'FAILED',qualification='installed-omp-kv-only-synthetic-subagent',
                    native=False,production_command=production_command,seconds=time.monotonic()-started,released=released,calls=calls,facts=facts,
                    errors=sorted(actor.error_codes),exit_code=actor.process.returncode,cleanup=cleanup,omp_sha256=digest(omp.resolve()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--omp',type=Path,required=True);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--production-command',action='store_true')
    result=run(**vars(parser.parse_args()));print(json.dumps(result,indent=2));raise SystemExit(result['verdict']!='PASSED')
