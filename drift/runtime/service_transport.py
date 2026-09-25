"""Loopback service transport and authenticated client."""
from __future__ import annotations
import json
import socket
import socketserver
from typing import Any
from drift.runtime.service_protocol import Op, ServiceError, json_safe, sign, verify
from drift.runtime.service_session import Session

MAX_LINE = 1 << 20                                          # a longer line closes the connection before authentication


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        session: Session = self.server.session          # type: ignore[attr-defined]
        secret: bytes = self.server.secret              # type: ignore[attr-defined]
        while True:
            line = self.rfile.readline(MAX_LINE + 1)
            if not line or len(line) > MAX_LINE or not line.endswith(b"\n"):
                return                                    # end of input, an oversize line or a cut-off one
            try:
                request = json.loads(line)
                authentic = isinstance(request, dict) and verify(request, secret)
            except Exception:                             # malformed, or a value the signature cannot encode
                return
            if not authentic:
                return                                    # close silently on auth failure; the run is untouched
            try:
                response = {"id": request.get("id"), "ok": True, "result": session.handle(request)}
            except ServiceError as error:
                response = {"id": request.get("id"), "ok": False, "error": error.code}
            except Exception:
                response = {"id": request.get("id"), "ok": False, "error": "INTERNAL"}
                if session.controller is not None:
                    session.controller.failed = True
            response = json_safe(response)
            response["auth"] = sign(response, secret)
            self.wfile.write(json.dumps(response, sort_keys=True).encode() + b"\n")
            self.wfile.flush()



class Service(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, port: int, secret: bytes, session: Session):
        if len(secret) < 32:
            raise ValueError("service secret must be at least 32 bytes")
        super().__init__(("127.0.0.1", port), _Handler)
        self.secret, self.session = secret, session



class Client:
    """Minimal authenticated client (tests and scripts)."""
    def __init__(self, port: int, secret: bytes, timeout: float = 30.0):
        self.secret = secret
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self.file = self.sock.makefile("rwb")
        self.next_id = 0

    def call(self, op: Op, **fields: Any) -> dict:
        self.next_id += 1
        request = {"id": self.next_id, "op": int(op), **fields}
        request["auth"] = sign(request, self.secret)
        self.file.write(json.dumps(request, sort_keys=True).encode() + b"\n")
        self.file.flush()
        line = self.file.readline()
        if not line:
            raise ConnectionError("service closed the connection")
        response = json.loads(line)
        if not verify(response, self.secret):
            raise ConnectionError("unauthenticated response")
        return response

    def close(self) -> None:
        self.sock.close()
