"""Bound the lifetime of only our own native worker, never the model server or handoffd."""
import argparse
import contextlib
import json
import math
import subprocess
import sys


def bounded(args):
    timeout = args.timeout
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 600:
        raise ValueError('TRANSCRIPT_BUDGET')
    try:
        result = subprocess.run([sys.executable, '-m', 'drift.transcript.process'],
                                input=json.dumps(vars(args), default=str), capture_output=True,
                                text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {'status': 'FAILED', 'error': 'TRANSCRIPT_WORKER_TIMEOUT',
                'server_request_cleanup': 'UNVERIFIED'}
    if result.returncode or len(result.stdout) > 2 * 1048576:
        return {'status': 'FAILED', 'error': 'TRANSCRIPT_WORKER_FAILED'}
    try:
        report = json.loads(result.stdout)
    except ValueError:
        return {'status': 'FAILED', 'error': 'TRANSCRIPT_WORKER_PROTOCOL'}
    if type(report) is not dict or report.get('status') not in ('PASSED', 'FAILED', 'BLOCKED', 'INVALID'):
        return {'status': 'FAILED', 'error': 'TRANSCRIPT_WORKER_PROTOCOL'}
    return report


def main():
    try:
        args = argparse.Namespace(**json.loads(sys.stdin.read(65537)))
        if args.command != 'transcript' or args.transcript_command not in ('capture', 'answer'):
            raise ValueError('TRANSCRIPT_OPERATION')
        from drift.transcript.cli import execute
        with contextlib.redirect_stdout(sys.stderr):
            report = execute(args)
    except (Exception, SystemExit) as error:
        kind = 'BLOCKED' if isinstance(error, ImportError) else 'FAILED'
        report = {'status': kind, 'error': 'TRANSCRIPT_NATIVE_DEPENDENCY' if kind == 'BLOCKED'
                  else 'TRANSCRIPT_NATIVE_OPERATION_FAILED'}
    print(json.dumps(report))


if __name__ == '__main__':
    main()
