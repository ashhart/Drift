"""Bound local SSH waits and retain remote cleanup acknowledgements for each session."""
import json
import math
import os
import selectors
import shlex
import subprocess
import time
from pathlib import Path

SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]


def remote_command(command, session, seconds, grace, lock=None):
    source = Path(__file__).with_name("remote_lease.py").read_text()
    spec = json.dumps(dict(command=command, session=session, seconds=seconds, grace=grace, lock=lock))
    return "exec python3 -c " + shlex.quote(source) + " " + shlex.quote(spec)


class LoopProcess:
    def __init__(self, argv, session, deadline, grace):
        self.session, self.deadline, self.grace = session, deadline, grace
        self.buffer, self.stopped, self.closed = bytearray(), None, False
        self.released = self.peer_announced = False
        self.child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.child.stdout, selectors.EVENT_READ)

    def readline(self, timeout):
        deadline = min(self.deadline, time.monotonic() + timeout)
        while True:
            if b"\n" in self.buffer:
                raw, _, self.buffer = self.buffer.partition(b"\n")
                event = raw.decode()
                try:
                    parsed = json.loads(event)
                except ValueError:
                    parsed = None
                if isinstance(parsed, dict) and "runner_stopped" in parsed:
                    if parsed["runner_stopped"] != self.session:
                        raise RuntimeError("wrong cleanup session")
                    self.stopped = parsed
                return event
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise TimeoutError("session output deadline exceeded")
            chunk = os.read(self.child.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("session output ended before expected event")
            self.buffer.extend(chunk)
            if len(self.buffer) > 4 * 1024 * 1024:
                raise RuntimeError("session output exceeds frame limit")

    def release(self):
        if self.released or self.closed or self.stopped is not None:
            raise RuntimeError("invalid session release")
        self.child.stdin.write(b"go\n")
        self.child.stdin.flush()
        self.released = True

    def peer_done(self):
        if not self.released or self.peer_announced or self.closed or self.stopped is not None:
            raise RuntimeError("invalid peer completion order")
        self.child.stdin.write(b"peer_done\n")
        self.child.stdin.flush()
        self.peer_announced = True

    def finish(self, timeout):
        deadline = min(self.deadline, time.monotonic() + timeout)
        while self.stopped is None:
            self.readline(max(0, deadline - time.monotonic()))
        self.child.wait(timeout=max(0, deadline - time.monotonic()))
        if self.child.returncode != 0 or self.stopped["reason"] != "completed" or self.stopped["returncode"] != 0:
            raise RuntimeError("remote session did not complete successfully")

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.child.stdin.close()
        deadline = time.monotonic() + 2 * self.grace + 3
        self.deadline = deadline
        try:
            while self.stopped is None:
                self.readline(max(0, deadline - time.monotonic()))
            self.child.wait(timeout=max(0, deadline - time.monotonic()))
        except (TimeoutError, RuntimeError, subprocess.TimeoutExpired):
            self.child.kill()
            self.child.wait(timeout=2)
        finally:
            self.selector.close()
            self.child.stdout.close()
        if self.stopped is None:
            raise RuntimeError("remote cleanup unconfirmed; remote lease remains the stop bound")


class LoopProcesses:
    def __init__(self, session, seconds=600, startup=60, grace=0.5):
        if not all(math.isfinite(n) and n > 0 for n in (seconds, startup, grace)):
            raise ValueError("process deadlines must be positive and finite")
        self.session, self.startup, self.grace = session, startup, grace
        self.deadline, self.processes, self.cleanup_errors = time.monotonic() + seconds, [], []

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("loop wall-time limit exceeded")
        return remaining

    def run(self, *argv, input=None):
        return subprocess.run(argv, input=input, capture_output=True, text=True, check=True,
                              timeout=min(self.startup, self.remaining())).stdout

    def spawn(self, host, command, *, lock=None):
        wrapped = remote_command(command, self.session, self.remaining(), self.grace, lock)
        process = LoopProcess(["ssh", *SSH_OPTIONS, host, wrapped], self.session, self.deadline, self.grace)
        self.processes.append(process)
        return process

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        for process in reversed(self.processes):
            try:
                process.close()
            except Exception as cleanup_error:
                self.cleanup_errors.append(str(cleanup_error))
        if self.cleanup_errors:
            raise RuntimeError("; ".join(self.cleanup_errors)) from error
