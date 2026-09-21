"""Loopback check of the dedicated Drift target on THIS machine only (127.0.0.1): real daemon, real wire protocol, the mailbox
producer over UDP and the consumer through the mapped region file. Starts the daemon, measures, stops it with SIGTERM."""
import argparse, importlib.util, json, os, signal, statistics, subprocess, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from drift.serving.mcdma_mailbox import RETRIES, MailboxError, MappedRegion, Reader, Writer

parser = argparse.ArgumentParser()
parser.add_argument("--build", type=Path, required=True)
parser.add_argument("--mib", type=int, default=16)
parser.add_argument("--rounds", type=int, default=200)
args = parser.parse_args()
spec = importlib.util.spec_from_file_location("drift_mcdma", args.build / "mcdma.py"); mcdma = importlib.util.module_from_spec(spec); spec.loader.exec_module(mcdma)
import ctypes, platform
mcdma._load = lambda: ctypes.CDLL(str(args.build / ("libmcdma.dylib" if platform.system() == "Darwin" else "libmcdma.so")))   # the library built for THIS port
with tempfile.TemporaryDirectory() as folder:
    region = str(Path(folder) / "region.bin")
    log = open(Path(folder) / "target.log", "w")                                # never an unread pipe: the daemon logs under a global lock
    daemon = subprocess.Popen([str(args.build / "drift_mcdma_target"), "127.0.0.1", str(args.mib), region], stdout=log, stderr=subprocess.STDOUT)
    try:
        refused = subprocess.run([str(args.build / "drift_mcdma_target"), "0.0.0.0", "16", region], capture_output=True, text=True)
        time.sleep(1.5)                                                          # upstream prints "ready" BEFORE its four workers bind their sockets
        reader = Reader(MappedRegion(region))
        conn = mcdma.open("127.0.0.1")
        assert conn.region_len == args.mib << 20, conn.region_len
        writer, report = Writer(conn), {"refuses_wildcard_bind": refused.returncode == 2, "region_mib": args.mib}
        for size in (4096, 94208, 188416, 1048576):                              # 94 KB ~ 16 tokens of GLM entries packed fp8; 188 KB ~ float16
            payload, publish, consume, total = os.urandom(size), [], [], []
            for _ in range(args.rounds if size < 1 << 20 else 40):
                t0 = time.perf_counter(); writer.publish(payload); t1 = time.perf_counter()
                got = reader.poll(); t2 = time.perf_counter()
                assert got == payload and writer.acknowledged(); t3 = time.perf_counter()
                publish.append(t1 - t0); consume.append(t2 - t1); total.append(t3 - t0)
            report[f"{size}_bytes"] = {k: round(statistics.median(v) * 1e3, 3) for k, v in (("publish_verified_ms", publish), ("consume_ms", consume), ("publish_to_ack_seen_ms", total))}
        writer.publish(b"unacknowledged")
        try:
            writer.publish(b"too early"); report["reuse_before_ack_refused"] = False
        except MailboxError:
            report["reuse_before_ack_refused"] = True
        Reader(MappedRegion(region))                                              # consumer restart: new session
        try:
            writer.acknowledged(); report["restart_detected"] = False
        except MailboxError:
            report["restart_detected"] = True
        conn.close()
    finally:
        daemon.send_signal(signal.SIGTERM)
        try:
            daemon.wait(timeout=5); stopped = "SIGTERM"
        except subprocess.TimeoutExpired:
            daemon.kill(); daemon.wait(); stopped = "SIGKILL (did not stop on SIGTERM)"
    report["daemon_stopped_by"] = stopped; report["transport_retries"] = RETRIES["count"]
print(json.dumps(report, indent=1))
