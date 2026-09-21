"""Minimal reproduction: one-sided MCDMA READs of certain sizes never receive their reply over the USB-C legs.

Usage (from the directory holding mcdma.py and the client library built for the target's port):
    python3 repro_read_size_fault.py <target-ip> [<source-ip>] [--port-lib <dir>]
Reads only, at offset 8 MiB of the target's region; nothing is written. Each failing size costs the client's 2 s timeout.
Observed 2026-09-21 Studio -> spark-a.invalid and Studio -> spark-b.invalid over USB-C: every size with size % 1024 in {675, 676, 677, 678} fails,
every other size passes; the same sizes pass when the client runs on the target's own host (no USB link in the path)."""
import ctypes, importlib.util, platform, sys, time
from pathlib import Path
args = [a for a in sys.argv[1:] if not a.startswith("--")]
here = Path(sys.argv[sys.argv.index("--port-lib") + 1]) if "--port-lib" in sys.argv else Path(".")
spec = importlib.util.spec_from_file_location("mcdma", here / "mcdma.py"); mcdma = importlib.util.module_from_spec(spec); spec.loader.exec_module(mcdma)
mcdma._load = lambda: ctypes.CDLL(str((here / ("libmcdma.dylib" if platform.system() == "Darwin" else "libmcdma.so")).resolve()))
conn = mcdma.open(args[0], src=args[1] if len(args) > 1 else None)
sizes = sorted(set(list(range(660, 700)) + [k * 1024 + r for k in (1, 2, 4, 7) for r in (674, 675, 676, 677, 678, 679)] + [64, 512, 1024, 4096, 8192, 8960]))
failed, t0 = [], time.time()
for n in sizes:
    try:
        assert len(conn.get(n, offset=8 << 20)) == n
    except Exception:
        failed.append(n)
conn.close()
print({"target": args[0], "sizes_tried": len(sizes), "failed": failed, "failed_mod_1024": sorted({n % 1024 for n in failed}), "seconds": round(time.time() - t0, 1)})
