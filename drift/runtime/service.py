"""Worker service CLI and stable public imports; see docs/reference/service/SERVICE_PROTOCOL.md."""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
from drift.runtime.service_protocol import Op, ServiceError, canonical_json, js_number, sign, verify
from drift.runtime.service_session import Session
from drift.runtime.service_transport import Client, Service


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--secret-env", default="DRIFT_SERVICE_SECRET")
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--builder", default="drift.runtime.builders:from_manifest")
    args = parser.parse_args()
    secret = os.environ.get(args.secret_env, "").encode()
    module, name = args.builder.split(":")
    builder = getattr(__import__(module, fromlist=[name]), name)
    args.audit_root.mkdir(parents=True, exist_ok=True)
    service = Service(args.port, secret, Session(builder, args.audit_root, hashlib.sha256(secret).digest()))
    service.serve_forever()



if __name__ == "__main__":
    main()
