"""Run only an explicitly selected real worker with one fixed non-I/O echo tool."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    owner = args.owner_config.resolve(strict=True)
    if digest(owner) != args.owner_sha256:
        raise ValueError('owner config hash mismatch')
    config = json.loads(owner.read_text())
    entry = next(worker for worker in config['workers'] if worker['identity']['model_id'] == args.model)
    limits = entry['limits']
    if entry['expected']['backend'] != 'live' or limits['max_turns'] != 2 or limits['max_output_tokens'] > 64 or limits['deadline_ms'] > 60000:
        raise ValueError('native probe requires live backend, exactly two turns, at most 64 new tokens per turn and 60 seconds')
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='drift-native-echo-') as directory:
        root = Path(directory).resolve()
        (root / 'agent').mkdir()
        facts_path = root / 'facts.json'
        env = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'PI_CODING_AGENT_DIR': str(root / 'agent'), 'PI_CONFIG_DIR': os.path.relpath(root, Path.home()), 'PI_NO_PTY': '1', 'TERM': 'dumb', 'NO_COLOR': '1', 'TMPDIR': str(root), 'DRIFT_WORKER_CONFIG': str(owner), 'DRIFT_WORKER_CONFIG_SHA256': args.owner_sha256, 'DRIFT_NATIVE_REPORT': str(facts_path)}
        here = Path(__file__).parent.resolve()
        command = [str(args.omp.resolve(strict=True)), '--cwd', str(root), '--no-session', '--no-tools', '--no-lsp', '--no-pty', '--no-extensions', '--no-skills', '--no-rules', '--no-title', '--no-prewalk', '--extension', str(here / 'experimental_extension.mjs'), '--extension', str(here / 'native_echo.mjs'), '--model', 'drift-experimental/' + args.model, '--thinking', 'off', '--print', '--mode', 'json', '--max-time', '60', 'Call drift_native_echo exactly once, then answer OK; do not call any other tools.']
        result = subprocess.run(command, cwd=root, env=env, capture_output=True, timeout=70)
        facts = json.loads(facts_path.read_text()) if facts_path.exists() else {}
        passed = result.returncode == 0 and facts.get('tool_calls') == 1 and facts.get('other_tools') == 0 and facts.get('assistant_turns') == 2 and 0 < facts.get('output_tokens', 0) <= 128
        return dict(verdict='PASSED' if passed else 'BLOCKED', model=args.model, memory_mode=entry['memory_mode'], native_state=entry['expected']['nativeStates'][0], exit_code=result.returncode, facts=facts, seconds=round(time.monotonic() - started, 3), owner_sha256=args.owner_sha256, executable_sha256=digest(args.omp.resolve()), stdout_sha256=hashlib.sha256(result.stdout).hexdigest(), stderr_sha256=hashlib.sha256(result.stderr).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--omp', type=Path, required=True)
    parser.add_argument('--owner-config', type=Path, required=True)
    parser.add_argument('--owner-sha256', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['verdict'] == 'PASSED' else 2)
