"""Stage an explicit provider boundary with private owner bindings and unchanged Duo."""
import json
from pathlib import Path
import shutil
import time
from duo_setup import prepare_duo
from registration_setup import digest
from provider_boundary_setup import FILES as BOUNDARY_FILES


FILES = (*BOUNDARY_FILES, 'experimental_extension.mjs')
FIXTURES = ('boundary_probe_fixture.py', 'declared_worker_fixture.py', 'provider_boundary_fixture.py', 'boundary_probe_observer.mjs', 'boundary_probe_after.mjs')


def write(path, value):
    path.write_text(json.dumps(value)); path.chmod(0o600)


def prepare(root, source, duo, candidate, provider_source, mode):
    task, control = root / 'task', root / 'control'
    task.mkdir(mode=0o700); control.mkdir(mode=0o700)
    original = prepare_duo(task, source, duo); scripts = task / 'scripts/omp'
    for name in FILES: shutil.copyfile(candidate / 'scripts/omp' / name, scripts / name)
    for name in FIXTURES: shutil.copyfile(Path(__file__).with_name(name), scripts / name)
    for path in (provider_source / 'plugin/omp-drift/src').glob('worker_*.ts'):
        shutil.copyfile(path, task / 'plugin/omp-drift/src' / path.name)
    config = json.loads(original.read_text())
    for entry in config['workers']:
        entry.update(memory_mode='linked', communication_mode='text_and_artifacts', session_binding='fixed')
        path = Path(entry['command']['args'][4]); worker = json.loads(path.read_text())
        worker.update(fixed_session=entry['identity']['session'], boundary_root=str(control)); write(path, worker)
        entry['command']['args'] = ['-S', '-m', 'scripts.omp.provider_boundary_fixture', '--config', str(path)]
        entry['artifacts'] += [dict(path=str(scripts / name), sha256='') for name in FIXTURES if name.endswith('.py')]
        for artifact in entry['artifacts']: artifact['sha256'] = digest(artifact['path'])
    owner = control / 'owner.json'; write(owner, config); original.unlink()
    (task / 'agent').mkdir()
    (task / 'agent/config.yml').write_text('compaction:\n  enabled: false\nextensionHandlers:\n  toolCallTimeoutMs: 150\n')
    now = int(time.time() * 1000)
    actors = [dict(role=role, model='drift-experimental/' + name, worker=name, session=name) for role, name in [('parent', 'glm-fixture'), ('child', 'qwen-fixture')]]
    boundary = dict(v=1, root=str(control), task_root=str(task), started_at_ms=now, deadline_ms=now + (3500 if mode == 'deadline' else 10000), nonce='a' * 32, actors=actors)
    path = control / 'config.json'; write(path, boundary)
    hashes = {str(path.relative_to(task)): digest(path) for path in sorted((task / 'plugin/omp-drift/src').glob('worker_*.ts'))}
    hashes.update({name: digest(scripts / name) for name in (*FILES, *FIXTURES)})
    return task, control, owner, path, boundary, hashes
