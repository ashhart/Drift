"""Require fresh owner-bound aggregate cleanup receipts and discard generated text."""
import hashlib
import json
import math
import subprocess


def cleanup_ok(value, *, max_seconds=60):
    if not isinstance(value, dict): return False
    if 'worker' in value:
        if value.get('status') != 'PASSED' or value.get('cleanup_confirmed') is not True or value.get('owner_thread_joined') is not True: return False
        value = value['worker']
    return isinstance(value, dict) and value.get('event') == 'worker_terminated' and value.get('reason') in ('input_eof', 'child_exit') and value.get('returncode') == 0 and value.get('child_reaped') is True and value.get('process_group_alive') is False and value.get('term_sent') is False and value.get('kill_sent') is False and type(value.get('wall_seconds')) in (int, float) and math.isfinite(value['wall_seconds']) and 0 <= value['wall_seconds'] <= max_seconds


def read_cleanup(item, *, fresh, max_seconds=60):
    spec = item['command']
    result = subprocess.run([spec['executable'], *spec['args']], cwd=spec['cwd'], env=spec['env'], capture_output=True, timeout=5)
    if len(result.stdout) > 8192 or len(result.stderr) > 8192: raise ValueError('ROOM_CLEANUP_SIZE')
    if fresh:
        if result.returncode != 3 or result.stdout or result.stderr: raise ValueError('ROOM_STALE_CLEANUP')
        return None
    if result.returncode != 0: raise ValueError('ROOM_MISSING_CLEANUP')
    value = json.loads(result.stdout)
    passed = cleanup_ok(value, max_seconds=max_seconds)
    actual = value.get('worker', value) if isinstance(value, dict) else {}
    safe = {key: actual.get(key) for key in ('returncode', 'child_reaped', 'process_group_alive', 'term_sent', 'kill_sent', 'wall_seconds', 'input_bytes', 'output_bytes') if type(actual.get(key)) in (int, float, bool)}
    return dict(worker=item['worker'], session=item['session'], passed=passed, receipt=safe, sha256=hashlib.sha256(result.stdout).hexdigest())


def room_ok(facts, selectors, token_limits=None, *, max_turns=6):
    if facts.get('errors') or facts.get('blocked') or facts.get('task_calls') != 1 or facts.get('hub_calls', 0) < 1 or facts.get('todo_calls', 0) < 1: return False
    workers = facts.get('workers', {})
    if set(workers) != set(selectors): return False
    limits = token_limits if token_limits is not None else dict.fromkeys(selectors, 20000)
    return set(limits) == set(selectors) and all(1 <= item.get('turns', 0) <= max_turns and 0 < item.get('output_tokens', 0) <= max_turns * 512 and 0 < item.get('input_tokens', 0) + item.get('output_tokens', 0) <= limits[selector] and item.get('shutdown') is True for selector, item in workers.items())
