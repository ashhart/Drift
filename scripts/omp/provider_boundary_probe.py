"""Qualify the installed OMP provider rendezvous using only a public synthetic Duo fixture."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from provider_probe_setup import prepare, write
from provider_boundary_verdict import assess
from provider_probe_process import ProbeProcess
from probe import probe_environment, sandbox_policy
from registration_setup import digest


def read(path, fallback=None):
    return json.loads(path.read_text()) if path.exists() else fallback


def gone(pid):
    if type(pid) is not int or pid <= 1: return False
    try: os.kill(pid, 0)
    except ProcessLookupError: return True
    return False


def run(omp, source, duo, candidate, mode, provider_source=None):
    with tempfile.TemporaryDirectory(prefix='drift-provider-boundary-') as directory:
        root = Path(directory).resolve(); root.chmod(0o700)
        task, control, owner, config_path, boundary, hashes = prepare(root, source, duo, candidate, provider_source or candidate, mode)
        scripts = task / 'scripts/omp'; events_path = task / 'events.jsonl'
        env = probe_environment(task) | dict(DRIFT_WORKER_CONFIG=str(owner), DRIFT_WORKER_CONFIG_SHA256=digest(owner), DRIFT_PROVIDER_BOUNDARY_CONFIG=str(config_path), DRIFT_PROVIDER_BOUNDARY_SHA256=digest(config_path), DRIFT_BOUNDARY_EVENTS=str(events_path))
        policy = root / 'policy.sb'; policy.write_text(sandbox_policy(root, Path.home()))
        command = ['/usr/bin/sandbox-exec', '-f', str(policy), str(omp.resolve()), '--cwd', str(task), '--no-session', '--tools', 'task,hub,todo', '--no-lsp', '--no-pty', '--no-extensions', '--no-skills', '--no-rules', '--no-title', '--no-prewalk']
        for extension in ('experimental_extension.mjs', 'duo_fixture.mjs', 'boundary_probe_observer.mjs', 'boundary_probe_after.mjs'): command += ['--extension', str(scripts / extension)]
        command += ['--model', 'drift-experimental/glm-fixture', '--thinking', 'off', '--max-time', '12']
        rpc = mode == 'abort'; prompt = 'Coordinate the deterministic fixture through stock Duo.'
        command += ['--mode', 'rpc'] if rpc else ['--print', '--mode', 'json', prompt]
        actor = ProbeProcess(command, task, env, rpc=rpc)
        started = time.monotonic(); deadline = started + 18
        parked = None; released = None; abort_sent = False
        try:
            if rpc: actor.send(dict(id='probe-prompt', type='prompt', message=prompt))
            while actor.process.poll() is None and time.monotonic() < deadline - 2:
                ready = [control / (role + '-ready.json') for role in ('parent', 'child')]
                if all(path.exists() for path in ready) and parked is None:
                    parked = dict(at=int(time.time() * 1000), clock=time.monotonic(), workers={name: read(task / (name + '-worker.json'), {}) for name in ('glm-fixture', 'qwen-fixture')})
                if parked and released is None and mode in ('release', 'timeout', 'invalid') and time.monotonic() - parked['clock'] >= (.5 if mode != 'release' else .05):
                    released = int(time.time() * 1000)
                    for index, role in enumerate(('parent', 'child')):
                        value = dict(v=1, action='resume', nonce='b' * 32 if mode == 'invalid' else boundary['nonce'], ready_sha256=digest(ready[index]), peer_ready_sha256=digest(ready[1-index]), exchange_sha256='c' * 64)
                        temporary = control / (role + '-release.tmp'); write(temporary, value); temporary.rename(control / (role + '-release.json'))
                if parked and rpc and not abort_sent:
                    actor.send(dict(id='probe-abort', type='abort')); abort_sent = True
                if rpc and actor.agent_end and (actor.abort_ack or not parked): actor.eof()
                time.sleep(.01)
        finally: cleanup = actor.close(deadline)
        if parked: parked.pop('clock')
        events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
        workers = {name: read(task / (name + '-worker.json'), {}) for name in ('glm-fixture', 'qwen-fixture')}
        cleanup['worker_pids_gone'] = len(workers) == 2 and all(worker.get('pgid') == actor.process.pid and gone(worker.get('pid')) for worker in workers.values())
        report = dict(qualification='synthetic-installed-omp-provider-boundary', activation_exchange=False, mode=mode, exit_code=actor.process.returncode, seconds=round(time.monotonic() - started, 3), parked=parked, released_at=released, events=events, workers=workers, facts=read(task / 'facts.json', {}), ready_remaining=[role for role in ('parent', 'child') if (control / (role + '-ready.json')).exists()], failed_roles=[role for role in ('parent', 'child') if (control / (role + '-failed.json')).exists()], abort_sent=abort_sent, abort_ack=actor.abort_ack, error_codes=sorted(actor.error_codes), cleanup=cleanup, executable_sha256=digest(omp.resolve()), candidate_hashes=hashes)
        report['boundary_evidence'] = {path.name: dict(sha256=digest(path), bytes=path.stat().st_size) for path in control.glob('*.json') if path.name != 'owner.json'}
        return report | assess(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('omp', 'source', 'duo', 'candidate'): parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--provider-source', type=Path)
    parser.add_argument('--mode', choices=['release', 'timeout', 'invalid', 'deadline', 'abort'], default='timeout')
    report = run(**vars(parser.parse_args())); print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['verdict'] == 'PASSED' else 2)
