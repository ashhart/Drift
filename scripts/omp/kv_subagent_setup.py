"""Stage a no-inference installed-OMP check with a pinned KV-only parent and child."""
import json
from pathlib import Path
import shutil
import sys
import time
from registration_setup import prepare, digest


def save(path, value):
    path.write_text(json.dumps(value)); path.chmod(0o600)


def stage(root, source, *, production_command=False):
    task, control = root / 'task', root / 'control'
    task.mkdir(mode=0o700); control.mkdir(mode=0o700)
    owner = prepare(task, source)
    if production_command:
        for path in (source / 'plugin/omp-drift/src').glob('*.ts'):
            shutil.copyfile(path, task / 'plugin/omp-drift/src' / path.name)
        for name in ('subagent_command.mjs', 'subagent_release.mjs'):
            shutil.copyfile(source / 'scripts/omp' / name, task / 'scripts/omp' / name)
    fixture = task / 'scripts/omp/kv_subagent_fixture.py'
    shutil.copyfile(Path(__file__).with_name(fixture.name), fixture)
    config = json.loads(owner.read_bytes())
    for role, entry in zip(('parent', 'child'), config['workers']):
        entry.update(memory_mode='linked', experimental_multi_turn=True,
                     communication_mode='kv_only', session_binding='fixed', subagent_role=role)
        entry['limits'].update(max_input_bytes=262144, max_session_tokens=32768, max_turns=6, deadline_ms=10000)
        entry['context_window'] = 32768
        command = entry['command']; command.update(executable=sys.executable, sha256=digest(sys.executable))
        path = Path(command['args'][4]); worker = json.loads(path.read_bytes())
        worker.update(fixture_role=role, limits=entry['limits'])
        save(path, worker)
        command['args'][-1] = 'scripts.omp.kv_subagent_fixture:KvFixture'
        entry['artifacts'].append(dict(path=str(fixture), sha256=digest(fixture)))
        for artifact in entry['artifacts']: artifact['sha256'] = digest(artifact['path'])
    owner.unlink(); owner = control / 'owner.json'; save(owner, config)
    (task / 'agent').mkdir()
    (task / 'agent/config.yml').write_text('compaction:\n  enabled: false\nasync:\n  enabled: true\ntools:\n  intentTracing: false\n')
    agents = task / '.omp/agents'; agents.mkdir(parents=True)
    (agents / 'drift-peer.md').write_text('---\nname: drift-peer\ndescription: KV-only fixture participant\nmodel: drift-experimental/qwen-fixture\ntools: drift_sync, yield\nblocking: false\nthinking-level: off\nprewalk: false\n---\nFollow your own preassigned Drift role.\n')
    actors = [dict(role=entry['subagent_role'], model='drift-experimental/'+entry['identity']['model_id'],
                   worker=entry['identity']['worker'], session=entry['identity']['session']) for entry in config['workers']]
    now = int(time.time()*1000)
    boundary = dict(v=2, max_epochs=2, root=str(control), task_root=str(task),
                    started_at_ms=now, deadline_ms=now+10000, nonce='a'*32, actors=actors)
    save(control / 'boundary.json', boundary)
    workers = []
    for index, entry in enumerate(config['workers']):
        identity = entry['identity']; peer = config['workers'][1-index]['identity']
        workers.append({key: identity[key] for key in ('worker','session','model_id')} | dict(
            route=entry['subagent_role'], exchange_session='fixture-exchange', source_worker=identity['worker'],
            target_worker=peer['worker'], ranks=['fixture-rank']) | (dict(delivery='next_turn_snapshot') if production_command else {}))
    exchange = dict(v=2 if production_command else 1, socket=str(control/'x.sock'), timeout_ms=1000, workers=workers)
    save(control/'exchange.json', exchange)
    return task, control, owner, boundary, exchange
