"""Serve private GLM own input and a separately pinned snapshot owner connection."""
import argparse
import json
import os
import sys
import time
from drift.serving.glm_owner_config import control_paths, create_owner
from drift.serving.glm_owner_input import OwnerInput
from drift.serving.glm_owner_socket import SnapshotSocket
from drift.serving.glm_restore_factory import factory
from drift.serving.glm_restore_route import PinnedRoute
from drift.serving.worker_owner_stdio import pinned_json
from drift.serving.worker_session import WorkerSession
from drift.serving.worker_stdio import serve


def serve_owner(configuration, *, backend_factory=factory, route_factory=PinnedRoute, input_fd=0, sink=None):
    started = time.monotonic()
    milliseconds = configuration['limits']['deadline_ms']
    if type(milliseconds) is not int or not 1000 <= milliseconds <= 180000: raise ValueError('OWNER_LIMIT')
    deadline = started + milliseconds / 1000
    socket_path, evidence_path = control_paths(configuration)
    evidence_fd = os.open(evidence_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    backend = bank = endpoint = session = None
    result = {'status': 'FAILED', 'control_failed': False, 'owner_thread_joined': False}
    try:
        backend, bank = create_owner(configuration, backend_factory=backend_factory, route_factory=route_factory)
        endpoint = SnapshotSocket(socket_path, deadline, bank)
        session = WorkerSession(configuration['worker'], configuration['pins'], configuration['limits'], backend,
                                allow_own_control=configuration.get('experimental_multi_turn', False))
        session.session = bank.session
        source = OwnerInput(input_fd, deadline-.5, endpoint.failed.is_set)
        endpoint.start(None)
        code = serve(session, source, sys.stdout if sink is None else sink)
        result['control_failed'] = endpoint.failed.is_set()
        if code == 0 and session.closed and not result['control_failed'] and time.monotonic() < deadline:
            result['status'] = 'PASSED'
        result['reason'] = source.reason or ('closed' if session.closed else 'own_protocol')
    except Exception:
        result.update(status='FAILED', reason='owner_failure')
    finally:
        if endpoint is not None:
            try: endpoint.close(); result['owner_thread_joined'] = True
            except Exception: result['status'] = 'FAILED'
            result['control_failed'] = endpoint.failed.is_set()
            result['owner_requests'] = endpoint.requests
        if backend is not None:
            try: backend.close()
            except Exception: result['status'] = 'FAILED'
            result['restoration'] = backend.restoration.report()
        if bank is not None:
            bank.close(); result['bank_poisoned'] = bank.failed
            result['versions'] = [version.receipt() for version in bank.versions]
        if time.monotonic() >= deadline or result['control_failed'] or not result['owner_thread_joined']:
            result['status'] = 'FAILED'
        result['wall_seconds'] = time.monotonic()-started
        raw = json.dumps(result, allow_nan=False).encode()+b'\n'
        try:
            if len(raw) > 1048576: raise ValueError('OWNER_EVIDENCE_LIMIT')
            if os.write(evidence_fd, raw) != len(raw): raise ValueError('OWNER_EVIDENCE_WRITE')
        finally: os.close(evidence_fd)
    return 0 if result['status'] == 'PASSED' else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args()
    try: return serve_owner(pinned_json(args.config, args.config_sha256))
    except Exception: return 2


if __name__ == '__main__': raise SystemExit(main())
