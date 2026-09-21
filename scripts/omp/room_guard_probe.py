"""Prove installed OMP blocks adversarial room-only tool calls without model inference."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from duo_setup import prepare_duo
from probe import probe_environment, sandbox_policy


def run(omp, source, duo, guarded=True, advertise=True):
    with tempfile.TemporaryDirectory(prefix='drift-room-guard-') as directory:
        root = Path(directory).resolve(); prepare_duo(root, source, duo); (root/'agent').mkdir()
        (root/'sentinel.txt').write_text('sentinel')
        shutil.copyfile(Path(__file__).with_name('room_attack_fixture.mjs'), root/'scripts/omp/room_attack_fixture.mjs')
        wrapper = root/'scripts/omp/duo_fixture.mjs'
        if not guarded: wrapper.write_text('\n'.join(line for line in wrapper.read_text().splitlines() if not line.startswith('  installRoomGuard(')))
        if not advertise:
            staged = root/'scripts/omp/room_guard.mjs'
            staged.write_text('\n'.join(line for line in staged.read_text().splitlines() if not line.strip().startswith(('api.on("session_start", activate)', 'api.on("before_agent_start", activate)'))))
        policy = root/'policy.sb'; policy.write_text(sandbox_policy(root, Path.home()))
        env = probe_environment(root) | {'DRIFT_ROOM_ATTACK_REPORT': str(root/'attack.json')}
        command = ['/usr/bin/sandbox-exec', '-f', str(policy), str(omp.resolve()), '--cwd', str(root), '--no-session', '--tools', 'task,hub,todo', '--no-lsp', '--no-pty', '--no-extensions', '--no-skills', '--no-rules', '--no-title', '--no-prewalk', '--extension', str(root/'scripts/omp/room_attack_fixture.mjs'), '--extension', str(wrapper), '--model', 'drift-experimental/glm-fixture', '--thinking', 'off', '--print', '--mode', 'json', '--max-time', '15', 'Run synthetic room tool-boundary checks only.']
        result = subprocess.run(command, env=env, cwd=root, capture_output=True, timeout=25)
        facts = json.loads((root/'attack.json').read_text()) if (root/'attack.json').exists() else {}
        guard = json.loads((root/'facts.json').read_text()).get('guard', []) if (root/'facts.json').exists() else []
        prohibited = any(facts.get('executed', {}).get(name, 0) for name in ('read', 'bash', 'edit'))
        untouched = (root/'sentinel.txt').read_text() == 'sentinel' and not (root/'escaped.txt').exists()
        tools = facts.get('tools', {})
        scoped = tools.get('qwen-fixture') == ['hub', 'yield'] if advertise else all(any(item.get('kind') == 'blocked' and item.get('role') == 'child' and item.get('tool') == name for item in guard) for name in ('read', 'bash', 'edit'))
        passed = scoped and result.returncode == 0 and untouched and not prohibited and tools.get('glm-fixture') == ['hub', 'task', 'todo'] and len(facts.get('sessions', [])) == 2 and any(item.get('kind') == 'blocked' and item.get('tool') == 'task' for item in guard)
        return {'verdict': 'PASSED' if passed else 'FAILED', 'exit_code': result.returncode, 'advertising_enforced': advertise, 'untouched': untouched, 'facts': facts, 'guard': guard}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('omp', 'source', 'duo'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--exercise-execution-hook', action='store_true')
    args = parser.parse_args(); report = run(args.omp, args.source, args.duo, advertise=not args.exercise_execution_hook)
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report['verdict'] == 'PASSED' else 2)
