"""Bind the public boundary fixture to its declared owner session."""
import argparse
import json
import os
import sys
from drift.serving.worker_artifacts import verify_worker_manifests
from drift.serving.worker_stdio import serve
from scripts.omp.boundary_probe_fixture import BoundaryFixture
from scripts.omp.declared_worker_fixture import FixedSession


class PublicBoundaryFixture(BoundaryFixture):
    def __init__(self, config):
        super().__init__(config)
        self.facts.update(pid=os.getpid(), pgid=os.getpgrp())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    with open(parser.parse_args().config) as stream: config = json.load(stream)
    verify_worker_manifests(config)
    return serve(FixedSession(config, PublicBoundaryFixture(config)), sys.stdin, sys.stdout)


if __name__ == '__main__': raise SystemExit(main())
