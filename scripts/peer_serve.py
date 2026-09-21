"""Run ONE member of a two-member session on this host, coupled to its peer over TCP.

Each host runs this with its own member name and role. The manifest is the same file
on both sides (pinned by sha256 in the log). Inputs are the member's own greedy
decode from its setup context; nothing but wire v2 frames crosses the link.

  PYTHONPATH=. python scripts/peer_serve.py --manifest run.json --member A --role server --port 47400 \
      --epochs 8 --secret-env DRIFT_LINK_SECRET --audit-root local/peer-A
  PYTHONPATH=. python scripts/peer_serve.py --manifest run.json --member B --role client --host 192.0.2.10 --port 47400 ...

Use loopback or an approved tunnel; the HMAC key authenticates but does not encrypt.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path
import torch
from drift.runtime.builders import from_manifest
from drift.runtime.peer import PeerLink, PeerRunner
from drift.transport.wire2 import FrameCodec2

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--member", required=True)
parser.add_argument("--role", choices=["server", "client"], required=True)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--secret-env", default="DRIFT_LINK_SECRET")
parser.add_argument("--audit-root", type=Path, required=True)
parser.add_argument("--timeout", type=float, default=120.0)
args = parser.parse_args()

secret = os.environ.get(args.secret_env, "").encode()
if len(secret) < 32:
    raise SystemExit(f"BLOCKED: set {args.secret_env} to at least 32 bytes on both hosts")
raw = args.manifest.read_bytes()
manifest = json.loads(raw)
workers = from_manifest(manifest)
if args.member not in workers:
    raise SystemExit(f"member {args.member} is not in the manifest")
worker = workers[args.member]
if len(workers) != 2:
    raise SystemExit("peer_serve couples exactly two members; use the service for a hive")

if args.role == "server":
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.settimeout(args.timeout)
    listener.bind((args.host, args.port))
    listener.listen(1)
    sock, peer_address = listener.accept()
    listener.close()
else:
    sock = socket.create_connection((args.host, args.port), timeout=args.timeout)
    peer_address = sock.getpeername()
runner = PeerRunner(worker, PeerLink(sock, FrameCodec2(hashlib.sha256(secret).digest()), timeout=args.timeout))
args.audit_root.mkdir(parents=True, exist_ok=True)

vocab = worker.adapter.decoder.model.config.vocab_size if hasattr(worker.adapter, "decoder") else None
setup = manifest.get("setup_text", {}).get(args.member, "")
ids = torch.tensor([b % vocab if vocab else b for b in setup.encode()][:64] or [1], dtype=torch.long)
generated, timings = [], []
started = time.perf_counter()
with torch.no_grad():
    for _ in range(args.epochs):
        t0 = time.perf_counter()
        report = runner.tick(ids)
        timings.append(time.perf_counter() - t0)
        token = int(report.output[-1].argmax())
        generated.append(token)
        ids = torch.tensor([token], dtype=torch.long)
runner.link.close()
record = {"member": args.member, "role": args.role, "peer": list(peer_address), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
          "epochs": runner.epoch, "wire_bytes_sent": runner.wire_bytes_sent, "wire_bytes_received": runner.wire_bytes_received,
          "epoch_seconds": timings, "wall_seconds": time.perf_counter() - started, "generated_ids": generated,
          "counters": vars(worker.counters), "poisoned": worker.poisoned}
(args.audit_root / f"peer-{args.member}.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({k: v for k, v in record.items() if k not in ("generated_ids", "epoch_seconds")}, indent=2))
