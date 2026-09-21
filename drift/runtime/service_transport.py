"""Loopback service transport and authenticated client."""
from __future__ import annotations
import json
import socket
import socketserver
from typing import Any
from drift.runtime.service_protocol import Op, ServiceError, sign, verify
from drift.runtime.service_session import Session


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        session: Session = self.server.session          # type: ignore[attr-defined]
        secret: bytes = self.server.secret              # type: ignore[attr-defined]
        for line in self.rfile:
            try:
                request = json.loads(line)
                if not isinstance(request, dict) or not verify(request, secret):
                    return                                # close silently on auth failure
                response = {"id": request.get("id"), "ok": True, "result": session.handle(request)}
            except ServiceError as error:
                response = {"id": request.get("id") if isinstance(request, dict) else None, "ok": False, "error": error.code}
            except (json.JSONDecodeError, UnicodeDecodeError):
                return
            except Exception:
                response = {"id": request.get("id"), "ok": False, "error": "INTERNAL"}
                if session.controller is not None:
                    session.controller.failed = True
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
