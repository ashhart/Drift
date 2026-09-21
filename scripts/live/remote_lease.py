"""Bound one trusted remote command and terminate only its owned process group."""
import fcntl
import json
import math
import os
import selectors
import signal
import subprocess
import sys
import time

GUARD = """import json,os,signal,subprocess,sys,time
command,fd = sys.argv[1],int(sys.argv[2])
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen(command, shell=True, preexec_fn=lambda: signal.signal(signal.SIGTERM, signal.SIG_DFL))
code = child.wait()
os.write(fd, json.dumps(code).encode() + b'\\n')
os.close(fd)
while True: time.sleep(3600)
"""


def supervise(spec):
    seconds, grace = spec["seconds"], spec["grace"]
    if not all(math.isfinite(n) and n > 0 for n in (seconds, grace)):
        raise ValueError("invalid process deadline")
    lock = child = None
    read_fd, write_fd = os.pipe()
    previous = {}
    code, reason = None, "startup_failed"

    def cancelled(signum, frame):
        raise InterruptedError("remote supervisor interrupted")

    try:
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            previous[sig] = signal.signal(sig, cancelled)
        if spec.get("lock"):
            lock = open(spec["lock"], "a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child = subprocess.Popen([sys.executable, "-c", GUARD, spec["command"], str(write_fd)],
                                 stdin=subprocess.PIPE, pass_fds=(write_fd,), start_new_session=True,
                                 preexec_fn=lambda: signal.signal(signal.SIGTERM, signal.SIG_IGN))
        os.close(write_fd)
        write_fd = None
        deadline, control_index = time.monotonic() + seconds, 0
        allowed = (b"go\n", b"peer_done\n")
        with selectors.DefaultSelector() as selector:
            selector.register(sys.stdin, selectors.EVENT_READ, "control")
            selector.register(read_fd, selectors.EVENT_READ, "exit")
            control = bytearray()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    reason = "lease_expired"
                    break
                ready = selector.select(min(remaining, 0.25))
                for key, _ in ready:
                    if key.data == "exit":
                        raw = os.read(read_fd, 64)
                        code = json.loads(raw) if raw else None
                        reason = "completed" if code is not None else "guard_failed"
                        return code if code is not None else 125
                    data = os.read(sys.stdin.fileno(), 64)
                    if not data:
                        reason = "controller_disconnected"
                        return 125
                    control.extend(data)
                    while b"\n" in control:
                        line, _, rest = control.partition(b"\n")
                        frame = line + b"\n"
                        if control_index >= len(allowed) or frame != allowed[control_index]:
                            reason = "invalid_control"
                            return 125
                        child.stdin.write(frame)
                        child.stdin.flush()
                        control_index += 1
                        control = bytearray(rest)
                    if control and (control_index >= len(allowed) or not allowed[control_index].startswith(control)):
                        reason = "invalid_control"
                        return 125
        return 124
    except InterruptedError:
        reason = "cancelled"
        return 125
    finally:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        if child is not None:
            # The unreaped guard reserves its process-group ID through both signals.
            os.killpg(child.pid, signal.SIGTERM)
            time.sleep(grace)
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=grace + 1)
            child.stdin.close()
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)
        if lock is not None:
            lock.close()
        print(json.dumps({"runner_stopped": spec["session"], "returncode": code, "reason": reason}), flush=True)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(supervise(json.loads(sys.argv[1])))
