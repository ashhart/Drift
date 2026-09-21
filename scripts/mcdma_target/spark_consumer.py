"""Mailbox consumer on a Spark (stdlib only): maps the dedicated target's region file, starts a session, consumes `--count`
publications and records for each its sequence, size, CRC and the local monotonic time it was consumed. Exits on its own."""
import argparse, json, sys, time, zlib
sys.path.insert(0, "/root/drift-live/mcdma-target")
from mcdma_mailbox import MappedRegion, Reader
parser = argparse.ArgumentParser()
parser.add_argument("--region", default="/dev/shm/drift-mailbox/region.bin")
parser.add_argument("--count", type=int, required=True)
parser.add_argument("--timeout", type=float, default=120)
parser.add_argument("--out", required=True)
args = parser.parse_args()
reader, rows, deadline = Reader(MappedRegion(args.region)), [], time.time() + args.timeout
print(json.dumps({"session": reader.session}), flush=True)
while len(rows) < args.count and time.time() < deadline:
    payload = reader.poll()
    if payload is None:
        time.sleep(0.00005)
        continue
    rows.append({"sequence": reader.expected - 1, "bytes": len(payload), "crc32": zlib.crc32(payload)})
json.dump({"session": reader.session, "consumed": rows}, open(args.out, "w"))
print(json.dumps({"consumed": len(rows)}), flush=True)
