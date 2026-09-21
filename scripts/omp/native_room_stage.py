"""Freeze a room-only scratch bundle separately from owner evidence and commands."""
import json
import os
from pathlib import Path
import shutil
from native_room_profile import digest
from provider_boundary_setup import copy_boundary_modules

PROMPT = 'Use stock Duo to coordinate this room-only check: create one todo, dispatch exactly one task to duo-peer named DuoPeer asking it to send the fixed message room-ready to Main through hub, exchange a short acknowledgement with DuoPeer, then finish; do not request file, shell, edit, other agents or other tools.'


def prepare(profile, evidence):
    evidence = evidence.resolve()
    if any((path/'.git').exists() for path in (evidence, *evidence.parents)): raise ValueError('ROOM_EVIDENCE_SCOPE')
    evidence.mkdir(mode=0o700)
    root = evidence/'scratch'; root.mkdir(mode=0o700); (root/'agent').mkdir()
    (root/'agent/config.yml').write_text('compaction:\n  enabled: false\n')
    here = Path(__file__).resolve().parent; repo = here.parents[1]
    target = root/'scripts/omp'; target.mkdir(parents=True)
    for name in ('experimental_extension.mjs', 'native_room_extension.mjs', 'room_guard.mjs', 'room_policy.mjs', 'room_diagnostics.mjs', 'room_hub.mjs'):
        shutil.copyfile(here/name, target/name)
    copy_boundary_modules(target)
    providers = root/'plugin/omp-drift/src'; providers.mkdir(parents=True)
    for path in (repo/'plugin/omp-drift/src').glob('worker_*.ts'): shutil.copyfile(path, providers/path.name)
    duo = Path(profile['duo']['path'])
    shutil.copytree(duo/'src', target/'stock_duo/src')
    for folder in (root/'agents', root/'.omp/agents'):
        folder.mkdir(parents=True); shutil.copyfile(duo/'agents/duo-peer.md', folder/'duo-peer.md')
    scope = dict(root=str(root), parent=profile['parent'], child=profile['child'], run=evidence.name)
    (root/'scope.json').write_text(json.dumps(scope))
    env = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'PI_CODING_AGENT_DIR': str(root/'agent'), 'PI_CONFIG_DIR': os.path.relpath(root, Path.home()), 'PI_NO_PTY': '1', 'TERM': 'dumb', 'NO_COLOR': '1', 'TMPDIR': str(root), 'DRIFT_WORKER_CONFIG': profile['owner']['path'], 'DRIFT_WORKER_CONFIG_SHA256': profile['owner']['sha256'], 'DRIFT_ROOM_SCOPE': str(root/'scope.json'), 'DRIFT_ROOM_REPORT': str(evidence/'facts.json')}
    command = [profile['omp']['path'], '--cwd', str(root), '--no-session', '--tools', 'task,hub,todo', '--no-lsp', '--no-pty', '--no-extensions', '--no-skills', '--no-rules', '--no-title', '--no-prewalk', '--extension', str(target/'experimental_extension.mjs'), '--extension', str(target/'native_room_extension.mjs'), '--model', profile['parent'], '--thinking', 'off', '--print', '--mode', 'json', '--max-time', '50', PROMPT]
    pins = {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob('*')) if path.is_file()}
    return dict(version=1, command=command, env=env, cwd=str(root), pins=pins, prompt_bytes=len(PROMPT.encode()))


def verify_stage(prepared):
    root = Path(prepared['cwd'])
    if prepared['prompt_bytes'] > 1024: raise ValueError('ROOM_PROMPT')
    if any(digest(root/path) != expected for path, expected in prepared['pins'].items()): raise ValueError('ROOM_STAGE_PIN')
