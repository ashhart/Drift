"""Bridge private stdio through an owned on-host supervisor with separate evidence."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def launch_stdio(command, evidence, *, wall_seconds=60, stop_timeout=2):
    if not command or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in (wall_seconds, stop_timeout)):
        raise ValueError('INVALID_LIMIT')
    reserve = min(.5, wall_seconds / 10)
    if wall_seconds > 180 or wall_seconds - reserve <= 3 * stop_timeout + .25:
        raise ValueError('INVALID_LIMIT')
    started = time.monotonic()
    evidence_fd = os.open(evidence, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    control_read, control_write = os.pipe()
    process = None
    try:
        args = [sys.executable, '-m', 'drift.serving.worker_supervisor', '--control-fd', str(control_read),
                '--evidence-fd', str(evidence_fd), '--input-fd', '0', '--output-fd', '1',
                '--wall-seconds', str(wall_seconds - reserve), '--stop-timeout', str(stop_timeout), '--', *command]
        process = subprocess.Popen(args, stderr=subprocess.DEVNULL, pass_fds=(control_read, evidence_fd), start_new_session=True)
        os.close(control_read); control_read = None
        os.close(evidence_fd); evidence_fd = None
        try:
            process.wait(timeout=max(0, started + wall_seconds - reserve / 2 - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.close(control_write); control_write = None
            process.wait(timeout=max(0, started + wall_seconds - time.monotonic()))
        if time.monotonic() - started >= wall_seconds:
            return 2
        path = Path(evidence)
        if path.stat().st_size > 4096:
            return 2
        result = json.loads(path.read_bytes())
        clean = (result.get('event') == 'worker_terminated' and result.get('reason') in {'child_exit', 'input_eof'}
                 and result.get('returncode') == 0 and result.get('child_reaped') is True
                 and result.get('process_group_alive') is False)
        return 0 if clean else 2
    finally:
        for fd in (control_read, control_write, evidence_fd):
            if fd is not None:
                os.close(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--wall-seconds', type=float, default=60)
    parser.add_argument('--stop-timeout', type=float, default=2)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = vars(parser.parse_args())
    if args['command'][:1] == ['--']:
        args['command'] = args['command'][1:]
    try:
        return launch_stdio(**args)
    except Exception:
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
