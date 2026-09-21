"""Serve activation-only commands on a separately inherited private Unix socket."""
import json
import os
import socket
import threading
from drift.serving.worker_activation import ActivationControl
from drift.serving.worker_contract import require


class ActivationServer:
    def __init__(self, config, gate, layouts, on_failure):
        fd = config['fd']
        require(type(fd) is int and fd >= 3, 'CAPABILITY')
        self.control = ActivationControl(config, gate, layouts)
        self.socket = socket.socket(fileno=os.dup(fd))
        require(self.socket.family == socket.AF_UNIX and self.socket.type == socket.SOCK_STREAM, 'CAPABILITY')
        os.close(fd)
        self.socket.settimeout(0.1)
        self.stopping = threading.Event()
        self.on_failure = on_failure
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def start(self):
        self.thread.start()

    def _serve(self):
        pending = bytearray()
        try:
            while not self.stopping.is_set():
                try:
                    block = self.socket.recv(4096)
                except socket.timeout:
                    continue
                require(bool(block), 'WORKER')
                pending.extend(block)
                while b'\n' in pending:
                    line, _, rest = pending.partition(b'\n'); pending = bytearray(rest)
                    require(len(line) <= 4096, 'LIMIT')
                    response = self.control.apply(json.loads(line))
                    self.socket.sendall(json.dumps(response).encode() + b'\n')
                require(len(pending) <= 4096, 'LIMIT')
        except Exception:
            if not self.stopping.is_set():
                self.control.gate.poison()
                self.on_failure()
                try: self.socket.sendall(b'{"v":1,"op":"error","code":"ACTIVATION"}\n')
                except OSError: pass
        finally:
            self.socket.close()

    def close(self):
        self.stopping.set()
        try: self.socket.shutdown(socket.SHUT_RDWR)
        except OSError: pass
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        require(not self.thread.is_alive(), 'LIMIT')
