"""Bound a controller process group while hashing and discarding model output."""
import hashlib
import os
import signal
import subprocess
import threading


def execute(command, *, cwd, env, timeout=54, max_bytes=2097152):
    child = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    counts = {}; oversized = threading.Event()
    def consume(name, stream):
        size = 0; digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8192), b''):
            size += len(chunk); digest.update(chunk)
            if size > max_bytes:
                oversized.set()
                try: os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError: pass
        counts[name] = dict(bytes=size, sha256=digest.hexdigest()); stream.close()
    threads = [threading.Thread(target=consume, args=(name, stream), daemon=True) for name, stream in [('stdout', child.stdout), ('stderr', child.stderr)]]
    for thread in threads: thread.start()
    timed_out = False
    try: child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try: os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try: os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            child.wait(timeout=2)
    try:
        os.killpg(child.pid, signal.SIGKILL)
        descendants_killed = True
    except ProcessLookupError: descendants_killed = False
    for thread in threads: thread.join(timeout=1)
    return dict(exit_code=child.returncode, descendants_killed=descendants_killed, timed_out=timed_out, output_limit=oversized.is_set(), output_joined=all(not thread.is_alive() for thread in threads), **counts)
