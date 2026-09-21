"""Exercise candidate boundary hooks through actual OMP and unchanged public Duo."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
from boundary_probe_verdict import assess
from duo_setup import prepare_duo
from probe import probe_environment, sandbox_policy
from registration_setup import digest


FILES = ('project_boundary_control.mjs', 'project_boundary_extension.mjs', 'project_boundary_files.mjs')


def read(path, fallback=None):
    return json.loads(path.read_text()) if path.exists() else fallback


def write(path, value):
    path.write_text(json.dumps(value))
    path.chmod(0o600)


def run(omp, source, duo, candidate, mode):
    with tempfile.TemporaryDirectory(prefix='drift-boundary-probe-') as directory:
        root = Path(directory).resolve()
        task, control = root / 'task', root / 'control'
        task.mkdir(mode=0o700)
        control.mkdir(mode=0o700)
        owner = prepare_duo(task, source, duo)
        scripts = task / 'scripts/omp'
        for name in (*FILES, 'paused_echo_files.mjs'):
            shutil.copyfile(candidate / 'scripts/omp' / name, scripts / name)
        for name in ('boundary_probe_fixture.py', 'boundary_probe_observer.mjs', 'boundary_probe_after.mjs'):
            shutil.copyfile(Path(__file__).with_name(name), scripts / name)
        config = read(owner)
        for worker in config['workers']:
            worker['command']['args'][-1] = 'scripts.omp.boundary_probe_fixture:BoundaryFixture'
            worker_path = Path(worker['command']['args'][4])
            payload = read(worker_path) | {'boundary_root': str(control)}
            write(worker_path, payload)
            worker['artifacts'].append({'path': str(scripts / 'boundary_probe_fixture.py'), 'sha256': ''})
            for artifact in worker['artifacts']:
                artifact['sha256'] = digest(artifact['path'])
        write(owner, config)
        (task / 'agent').mkdir()
        timeout = 150 if mode == 'timeout' else 5000
        (task / 'agent/config.yml').write_text(f'compaction:\n  enabled: false\nextensionHandlers:\n  toolCallTimeoutMs: {timeout}\n')
        now = int(time.time() * 1000)
        actors = [dict(role=role, model='drift-experimental/' + name, worker=name, session=name) for role, name in [('parent', 'glm-fixture'), ('child', 'qwen-fixture')]]
        boundary = dict(v=1, root=str(control), task_root=str(task), started_at_ms=now, deadline_ms=now + (2500 if mode == 'deadline' else 10000), nonce='a' * 32, actors=actors)
        path = control / 'config.json'
        write(path, boundary)
        policy = root / 'policy.sb'
        policy.write_text(sandbox_policy(root, Path.home()))
        events_path = task / 'events.jsonl'
        env = probe_environment(task) | dict(DRIFT_WORKER_CONFIG=str(owner), DRIFT_WORKER_CONFIG_SHA256=digest(owner), DRIFT_PROJECT_BOUNDARY_CONFIG=str(path), DRIFT_PROJECT_BOUNDARY_SHA256=digest(path), DRIFT_BOUNDARY_EVENTS=str(events_path))
        command = ['/usr/bin/sandbox-exec', '-f', str(policy), str(omp.resolve()), '--cwd', str(task), '--no-session', '--tools', 'task,hub,todo', '--no-lsp', '--no-pty', '--no-extensions', '--no-skills', '--no-rules', '--no-title', '--no-prewalk']
        for extension in ('experimental_extension.mjs', 'duo_fixture.mjs', 'boundary_probe_observer.mjs', 'project_boundary_extension.mjs', 'boundary_probe_after.mjs'):
            command += ['--extension', str(scripts / extension)]
        command += ['--model', 'drift-experimental/glm-fixture', '--thinking', 'off', '--print', '--mode', 'json', '--max-time', '12', 'Coordinate the deterministic fixture through stock Duo.']
        process = subprocess.Popen(command, env=env, cwd=task, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        started = time.monotonic()
        parked = None
        released = None
        killed = False
        while process.poll() is None and time.monotonic() - started < 18:
            ready = [control / (role + '-ready.json') for role in ('parent', 'child')]
            if all(item.exists() for item in ready) and parked is None:
                parked = dict(at=int(time.time() * 1000), workers={name: read(task / (name + '-worker.json')) for name in ('glm-fixture', 'qwen-fixture')})
            if parked and released is None and mode in ('release', 'timeout', 'invalid'):
                time.sleep(0.05 if mode == 'release' else 0.5)
                for index, role in enumerate(('parent', 'child')):
                    write(control / (role + '-release.json'), dict(v=1, action='resume', nonce='b' * 32 if mode == 'invalid' else boundary['nonce'], ready_sha256=digest(ready[index]), peer_ready_sha256=digest(ready[1-index]), exchange_sha256='c' * 64))
                released = int(time.time() * 1000)
            if parked and mode == 'cancel' and not killed:
                os.killpg(process.pid, signal.SIGTERM)
                killed = True
            time.sleep(0.01)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            killed = True
        stdout, stderr = process.communicate(timeout=3)
        events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
        workers = {name: read(task / (name + '-worker.json'), {}) for name in ('glm-fixture', 'qwen-fixture')}
        report = dict(mode=mode, exit_code=process.returncode, seconds=round(time.monotonic()-started, 3), parked=parked, released_at=released, events=events, workers=workers, facts=read(task / 'facts.json', {}), ready_remaining=[role for role in ('parent', 'child') if (control / (role + '-ready.json')).exists()], killed=killed, executable_sha256=digest(omp.resolve()), candidate_hashes={name:digest(candidate / 'scripts/omp' / name) for name in FILES}, stdout_bytes=len(stdout), stderr_bytes=len(stderr))
        return report | assess(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('omp', 'source', 'duo', 'candidate'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--mode', choices=['release', 'timeout', 'invalid', 'deadline', 'cancel'], default='release')
    report = run(**vars(parser.parse_args()))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['verdict'] == 'PASSED' else 2)
