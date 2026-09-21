"""Prepare a task-local API smoke and print an explicit bounded OMP invocation."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
from api_fixture import API, VERIFY


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root, python):
    root.mkdir(parents=True, exist_ok=False)
    (root / 'api.py').write_text(API)
    (root / 'verify.py').write_text(VERIFY)
    config = dict(root=str(root), max_file_bytes=16384, max_output_bytes=4096, max_run_ms=5000, edit_paths=['api.py'], read_only_paths=['verify.py'],
        runs=[dict(name='verify', executable=str(python), sha256=digest(python), args=['-S', 'verify.py'], network_loopback=True, artifacts=[dict(path=str(root / 'verify.py'), sha256=digest(root / 'verify.py'))], runtime_read_roots=['/System', '/usr', '/Library/Frameworks/Python.framework/Versions/3.11'])])
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--scope-output', type=Path, required=True)
    parser.add_argument('--owner-config', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--omp', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=Path('/Library/Frameworks/Python.framework/Versions/3.11/bin/python3'))
    args = parser.parse_args()
    root = args.task_root.resolve()
    scope = args.scope_output.resolve()
    if scope.is_relative_to(root):
        raise ValueError('scope manifest must remain outside model-editable task directory')
    owner = args.owner_config.resolve(strict=True)
    workers = json.loads(owner.read_text())['workers']
    selected = next(worker for worker in workers if worker['identity']['model_id'] == args.model)
    limits = selected['limits']
    if selected['expected']['backend'] != 'live' or selected['memory_mode'] != 'no-link':
        raise ValueError('development smoke requires explicit live no-link owner configuration')
    if limits['max_turns'] > 6 or limits['max_output_tokens'] > 1024 or limits['max_session_tokens'] > 20000 or limits['deadline_ms'] > 120000 or selected['context_window'] > 16384:
        raise ValueError('smoke requires at most 6 turns, 1024 output tokens per turn, 20000 total tokens, 120 seconds and 16384 context tokens')
    config = prepare(root, args.python.resolve(strict=True))
    scope.write_text(json.dumps(config))
    here = Path(__file__).parent.resolve()
    command = [str(args.omp.resolve()), '--cwd', str(root), '--no-session', '--no-tools', '--no-lsp', '--no-pty', '--no-extensions', '--no-skills', '--no-rules', '--no-title', '--no-prewalk', '--extension', str(here / 'experimental_extension.mjs'), '--extension', str(here / 'task_tools.mjs'), '--model', 'drift-experimental/' + args.model, '--thinking', 'off', '--print', '--mode', 'json', '--max-time', '120', 'Read api.py and public verify.py; fix GET /health to return HTTP 200 with JSON {"status":"ok"}, preserve the JSON 404 response for missing paths, and run verify using the assigned task tools; edit only api.py.']
    environment = dict(DRIFT_WORKER_CONFIG=str(owner), DRIFT_WORKER_CONFIG_SHA256=digest(owner), DRIFT_TASK_CONFIG=str(scope), DRIFT_TASK_CONFIG_SHA256=digest(scope), PI_CODING_AGENT_DIR=str(scope.parent / 'omp-agent'), PI_CONFIG_DIR=str(scope.parent), PI_NO_PTY='1')
    print(json.dumps(dict(verdict='PREPARED_NOT_RUN', task_root=str(root), environment=environment, command=shlex.join(command)), indent=2))


if __name__ == '__main__':
    main()
