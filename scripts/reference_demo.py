"""Synthetic affine-map + framed-transfer smoke test, not a language experiment."""
import argparse
import json
from pathlib import Path
from uuid import UUID
import torch
from drift.core.types import Delta, KV
from drift.core.projector import BridgeProjector
from drift.core.memory import ForeignKVBank
from drift.transport.wire import FrameCodec
from drift.transport.backends import InProcTransport

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.out.exists(): parser.error("output exists")
torch.set_num_threads(1); torch.manual_seed(91)
src = KV(torch.randn(256, 2, 4), torch.randn(256, 2, 4))
wk, wv = torch.randn(8, 6), torch.randn(8, 6)
dst = KV((src.k.flatten(1) @ wk + 0.5).reshape(256, 1, 6),
         (src.v.flatten(1) @ wv - 0.2).reshape(256, 1, 6))
projector = BridgeProjector((2, 4), (1, 6)); projector.fit(src, dst, ridge=1e-6)
heldout = KV(torch.randn(16, 2, 4), torch.randn(16, 2, 4))
expected_k = (heldout.k.flatten(1) @ wk + 0.5).reshape(16, 1, 6)
session = UUID(int=91); frame = Delta(session, 0, 0, 0, 0, {0: heldout})
codec = FrameCodec(b"reference-demo-only-key-not-for-real-runs")
transport = InProcTransport(codec); transport.send(frame)
bank = ForeignKVBank(session, 0, [(0, 0, projector)], sinks=2, recent=32)
bank.commit(transport.receive())
error = float((bank.pin().layers[0].k - expected_k).abs().max())
report = {"fixture": "synthetic_affine_mapping", "max_abs_key_error": error,
          "status": "PASSED" if error < 1e-4 else "FAILED",
          "frame_bytes_including_hmac": len(codec.encode(frame)),
          "tensor_payload_bytes_float32": sum(x.k.numel() * 8 for x in frame.layers.values()),
          "receiver_resident_slots": bank.pin().positions.numel(),
          "cross_family_language_transfer": "NOT_EVALUATED",
          "mcdma": "NOT_USED", "energy_joules": None}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["status"] == "PASSED" else 1)
