"""Bound and reap the installed OMP probe while retaining only aggregate stream evidence."""
import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time


class ProbeProcess:
    def __init__(self, command, cwd, env, rpc=False):
        self.process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.PIPE if rpc else subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        self.counts = {}; self.output_limit = False; self.terminated = False
        self.error_codes = set(); self.abort_ack = False; self.agent_end = False; self.rpc = rpc
        self.threads = [threading.Thread(target=self._consume, args=(name, stream), daemon=True) for name, stream in [('stdout', self.process.stdout), ('stderr', self.process.stderr)]]
        for thread in self.threads: thread.start()

    def _consume(self, name, stream):
        size = 0; digest = hashlib.sha256(); pending = b''
        while chunk := os.read(stream.fileno(), 8192):
            size += len(chunk); digest.update(chunk)
            self.error_codes.update(match.decode() for match in re.findall(rb"(?:DRIFT_WORKER_[A-Z_]+|\[(?:context|stream|close|unknown):[A-Z_]+\])", chunk))
            if size > 2097152:
                self.output_limit = True; self._signal(signal.SIGTERM); break
            if self.rpc and name == 'stdout':
                pending += chunk
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    try: value = json.loads(line)
                    except (ValueError, UnicodeDecodeError): continue
                    if not isinstance(value, dict): continue
                    if value.get('type') == 'agent_end': self.agent_end = True
                    if value.get('type') == 'response' and value.get('command') == 'abort' and value.get('id') == 'probe-abort' and value.get('success') is True: self.abort_ack = True
        self.counts[name] = dict(bytes=size, sha256=digest.hexdigest()); stream.close()

    def send(self, value):
        self.process.stdin.write(json.dumps(value).encode() + b'\n'); self.process.stdin.flush()

    def eof(self):
        if self.process.stdin and not self.process.stdin.closed: self.process.stdin.close()

    def _signal(self, value):
        self.terminated = True
        try: os.killpg(self.process.pid, value)
        except ProcessLookupError: pass

    def close(self, deadline):
        self.eof()
        if self.process.poll() is None:
            self._signal(signal.SIGTERM)
            try: self.process.wait(timeout=max(.01, min(.5, deadline - time.monotonic())))
            except subprocess.TimeoutExpired: self._signal(signal.SIGKILL)
        try: self.process.wait(timeout=max(.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired: pass
        try: os.killpg(self.process.pid, 0)
        except ProcessLookupError: group_gone = True
        else:
            self._signal(signal.SIGKILL); group_gone = False
            while time.monotonic() < deadline:
                try: os.killpg(self.process.pid, 0)
                except ProcessLookupError: group_gone = True; break
                time.sleep(.01)
        for thread in self.threads: thread.join(timeout=max(0, deadline - time.monotonic()))
        return dict(reaped=self.process.poll() is not None, joined=all(not thread.is_alive() for thread in self.threads), group_gone=group_gone, terminated=self.terminated, output_limit=self.output_limit, **self.counts)
