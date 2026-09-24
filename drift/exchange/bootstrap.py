"""Run one private native coordinator lease without starting shared transport daemons."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import stat
import sys
import time

from drift.exchange.native_owners import NativeOwnerCoordinator
from drift.serving.worker_activation_files import private_root


def load_profile(path, expected):
    path = Path(path)
    private_root(path.parent)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError('BOOTSTRAP_PROFILE')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or not 0 < info.st_size <= 65536:
            raise ValueError('BOOTSTRAP_PROFILE')
        raw = stream.read(65537)
    if len(raw) != info.st_size or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('BOOTSTRAP_PIN')
    return json.loads(raw)


def serve(profile, expected, seconds, incoming, outgoing, *, factory=NativeOwnerCoordinator):
    if type(seconds) not in (int, float) or not 0 < seconds <= 170:
        raise ValueError('BOOTSTRAP_BUDGET')
    deadline = time.monotonic() + seconds
    controller = factory(load_profile(profile, expected), deadline)
    raw, stopping = bytearray(), False
    handlers = {}
    def stop(_number, _frame):
        nonlocal stopping
        stopping = True
    try:
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            handlers[number] = signal.signal(number, stop)
        controller.start()
        outgoing.write(json.dumps(dict(op='ready', profile_sha256=expected)) + '\n'); outgoing.flush()
        while not stopping:
            if controller.server.failed.is_set() or time.monotonic() >= deadline:
                raise ValueError('BOOTSTRAP_DEADLINE_OR_FAILURE')
            readable, _, _ = select.select([incoming], [], [], min(.05, max(0, deadline-time.monotonic())))
            if not readable:
                continue
            block = os.read(incoming.fileno(), 257-len(raw))
            if not block:
                if raw:
                    raise ValueError('BOOTSTRAP_CONTROL')
                break
            raw.extend(block)
            if len(raw) > 256:
                raise ValueError('BOOTSTRAP_CONTROL')
            if b'\n' in raw:
                if raw.count(b'\n') != 1 or not raw.endswith(b'\n') or json.loads(raw) != {'op': 'shutdown'}:
                    raise ValueError('BOOTSTRAP_CONTROL')
                break
    finally:
        try:
            controller.close()
        finally:
            for number, handler in handlers.items():
                signal.signal(number, handler)
    outgoing.write(json.dumps(dict(op='closed', profile_sha256=expected)) + '\n'); outgoing.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--seconds', type=float, default=120)
    args = parser.parse_args()
    serve(args.profile, args.sha256, args.seconds, sys.stdin.buffer, sys.stdout)


if __name__ == '__main__':
    main()
