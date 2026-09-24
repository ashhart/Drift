"""Studio producer: publish the same payload to BOTH ranks' dedicated targets over the two USB-C MCDMA legs, in parallel, and
wait for both acknowledgements. Measures per rank and for the pair. CPU only; opens no verbs objects."""
import argparse, ctypes, importlib.util, json, os, statistics, sys, threading, time, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from drift.serving.mcdma_links import parse_links
from drift.serving.mcdma_mailbox import RETRIES, Writer
parser = argparse.ArgumentParser()
parser.add_argument("--build", type=Path, required=True)
parser.add_argument("--rounds", type=int, default=100)
parser.add_argument("--sizes", type=int, nargs="+", default=[4096, 94208, 188416, 1048576])
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--links", required=True, help="MCDMA legs to each rank as target/source, head first")
args = parser.parse_args()
spec = importlib.util.spec_from_file_location("drift_mcdma", args.build / "mcdma.py"); mcdma = importlib.util.module_from_spec(spec); spec.loader.exec_module(mcdma)
mcdma._load = lambda: ctypes.CDLL(str(args.build / "libmcdma.dylib"))
LEGS = parse_links(args.links)
conns = {rank: mcdma.open(peer, src=src) for rank, (peer, src) in LEGS.items()}
writers = {rank: Writer(conn) for rank, conn in conns.items()}
report = {"sessions": {r: w.session for r, w in writers.items()}, "region_mib": {r: c.region_len >> 20 for r, c in conns.items()}, "sizes": {}, "sent": []}


def deliver(rank, payload, result):
    t0 = time.perf_counter(); writers[rank].publish(payload); t1 = time.perf_counter()
    while not writers[rank].acknowledged():
        pass
    result[rank] = (t1 - t0, time.perf_counter() - t0)


for size in args.sizes:
    per_rank, pair = {r: {"publish": [], "acknowledged": []} for r in LEGS}, []
    for _ in range(args.rounds if size < 1 << 20 else max(10, args.rounds // 5)):
        payload, result = os.urandom(size), {}
        threads = [threading.Thread(target=deliver, args=(rank, payload, result)) for rank in LEGS]
        t0 = time.perf_counter()
        for t in threads: t.start()
        for t in threads: t.join()
        pair.append(time.perf_counter() - t0); report["sent"].append({"bytes": size, "crc32": zlib.crc32(payload)})
        for rank, (p, a) in result.items():
            per_rank[rank]["publish"].append(p); per_rank[rank]["acknowledged"].append(a)
    ms = lambda v: {"median_ms": round(statistics.median(v) * 1e3, 3), "p95_ms": round(sorted(v)[int(len(v) * 0.95) - 1] * 1e3, 3), "max_ms": round(max(v) * 1e3, 3)}
    report["sizes"][str(size)] = {"rounds": len(pair), "both_ranks_acknowledged": ms(pair), **{rank: {k: ms(v) for k, v in d.items()} for rank, d in per_rank.items()}}
    print(size, json.dumps(report["sizes"][str(size)]), flush=True)
report["transport_retries"] = RETRIES["count"]
for c in conns.values(): c.close()
args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=1))
print(json.dumps({"published": len(report["sent"]), "transport_retries": RETRIES["count"]}))
