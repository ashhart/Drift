"""Use the real worker core while independently checking the fixture's fixed owner session."""
import argparse
import json
import sys
from drift.serving.worker_session import WorkerSession
from drift.serving.worker_stdio import serve
from drift.serving.worker_artifacts import verify_worker_manifests
from drift.serving.worker_contract import require
from scripts.omp.duo_backend_fixture import DuoFixture


class FixedSession(WorkerSession):
    def __init__(self, config, backend):
        super().__init__(config['worker'], config['pins'], config['limits'], backend, allow_own_control=True)
        self.expected_session = config['fixed_session']

    def _admit(self, frame):
        require(frame.get('session') == self.expected_session)
        super()._admit(frame)
        self.backend.facts['fixed_session_verified'] = True
        self.backend.facts['activation_operations'] = 0


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True); args = parser.parse_args()
    with open(args.config) as handle: config = json.load(handle)
    verify_worker_manifests(config)
    return serve(FixedSession(config, DuoFixture(config)), sys.stdin, sys.stdout)


if __name__ == '__main__': raise SystemExit(main())
