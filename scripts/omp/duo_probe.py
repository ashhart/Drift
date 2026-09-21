"""Qualify stock Duo child provider inheritance without model inference or network."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time
from probe import probe_environment, sandbox_policy
from registration_setup import digest
from duo_setup import prepare_duo


def run(omp, source, duo, *, setup=prepare_duo):
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='drift-duo-') as directory:
        root = Path(directory).resolve()
        owner = setup(root, source, duo)
        (root / 'agent').mkdir()
        policy = root / 'policy.sb'
        policy.write_text(sandbox_policy(root, Path.home()))
        env = probe_environment(root) | {'DRIFT_WORKER_CONFIG': str(owner), 'DRIFT_WORKER_CONFIG_SHA256': digest(owner)}
        command = ['/usr/bin/sandbox-exec', '-f', str(policy), str(omp.resolve()), '--cwd', str(root), '--no-session', '--tools', 'task,hub,todo', '--no-lsp', '--no-pty', '--no-extensions', '--no-skills', '--no-rules', '--no-title', '--no-prewalk', '--extension', str(root / 'scripts/omp/experimental_extension.mjs'), '--extension', str(root / 'scripts/omp/duo_fixture.mjs'), '--model', 'drift-experimental/glm-fixture', '--thinking', 'off', '--print', '--mode', 'json', '--max-time', '20', 'Coordinate the deterministic fixture through stock Duo.']
        result = subprocess.run(command, env=env, cwd=root, capture_output=True, timeout=30)
        facts = {}
        for name in ('facts', 'glm-fixture-worker', 'qwen-fixture-worker'):
            path = root / (name + '.json')
            facts[name] = json.loads(path.read_text()) if path.exists() else {}
        parent, child = facts['glm-fixture-worker'], facts['qwen-fixture-worker']
        tools_scoped = parent.get('advertised_tools') == ['hub', 'task', 'todo'] and child.get('advertised_tools') == ['hub', 'yield']
        passed = tools_scoped and result.returncode == 0 and 4 <= parent.get('stream', 0) <= 6 and 2 <= child.get('stream', 0) <= 6 and parent.get('errors') == 0 and child.get('errors') == 0 and parent.get('has_task') and parent.get('has_todo') and not child.get('has_task') and not child.get('has_todo') and parent.get('open') == child.get('open') == parent.get('close') == child.get('close') == 1 and not facts['facts'].get('errors') and parent.get('own_prompt_sha256') != child.get('own_prompt_sha256')
        return dict(verdict='PASSED' if passed else 'BLOCKED', exit_code=result.returncode, facts=facts, seconds=round(time.monotonic() - started, 3), executable_sha256=digest(omp.resolve()), duo_command_sha256=digest(duo / 'src/command.ts'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--omp', type=Path, required=True)
    parser.add_argument('--worker-source', type=Path, required=True)
    parser.add_argument('--duo', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.omp, args.worker_source, args.duo)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['verdict'] == 'PASSED' else 2)
