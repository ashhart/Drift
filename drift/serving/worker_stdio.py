"""Run a private bounded NDJSON worker over caller-owned standard input and output."""
import argparse
import importlib
import json
import sys
import threading
from drift.serving.worker_session import WorkerSession


def serve(session, source, sink):
    output_lock = threading.Lock()
    active = None

    def emit(frame):
        for response in session.handle(frame):
            with output_lock:
                sink.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + '\n')
                sink.flush()

    try:
        while True:
            cap = session.ceilings['max_input_bytes'] + 1024
            line = source.readline(cap + 1)
            if not line:
                break
            if len(line.encode('utf-8')) > cap or not line.endswith('\n'):
                return 2
            try:
                frame = json.loads(line)
            except (ValueError, TypeError):
                return 2
            if type(frame) is not dict:
                return 2
            if frame.get('op') == 'stream':
                if active is not None and session.active_seq is None:
                    active.join(timeout=1)          # the last turn has ended; its thread may still be finishing
                if active is not None and active.is_alive():
                    emit(frame)
                    return 2
                session.stream_started.clear()
                active = threading.Thread(target=emit, args=(frame,), daemon=True)
                active.start()
                if not session.stream_started.wait(timeout=1):
                    return 2
            else:
                emit(frame)
            if session.closed or session.poisoned:
                break
        return 0 if session.closed else 2
    finally:
        if active is not None and active.is_alive():
            session.cancel_requested.set()
            try: session.backend.cancel(session.active_seq)
            except Exception: pass
        try: session.backend.close()
        except Exception: pass
        if active is not None:
            active.join(timeout=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--backend', required=True)
    args = parser.parse_args()
    try:
        with open(args.config, encoding='utf-8') as handle:
            config = json.load(handle)
        from drift.serving.worker_artifacts import verify_worker_manifests
        verify_worker_manifests(config)
        module, name = args.backend.split(':', 1)
        backend = getattr(importlib.import_module(module), name)(config)
        session = WorkerSession(config['worker'], config['pins'], config['limits'], backend,
                                allow_own_control=config.get('experimental_multi_turn', False))
        return serve(session, sys.stdin, sys.stdout)
    except Exception:
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
