"""Safety guard for anything that loads a large model on the shared Mac Studio (oMLX runtime).

Why it exists: on 2026-09-19 a 132 GB Drift process drove the Studio to 0.13 GB free (jetsam event), and jobs were
started without looking at what else was resident; workers could also outlive a sleeping laptop and keep ~110 GB.
Every Studio-side script calls `acquire()` BEFORE loading weights:
  1. one model-holding Drift process at a time (exclusive lock; a second one refuses instead of doubling memory);
  2. a preflight check that enough memory is reclaimable, leaving a reserve for the owner's own workloads;
  3. hard MLX limits: a memory limit and a small buffer-cache limit, so the process fails instead of starving the host.
`idle_exit()` ends a worker that has had no command for a while (dropped ssh link, sleeping laptop)."""
from __future__ import annotations
import fcntl
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

LOCK = Path("~/drift/local/studio/.model.lock").expanduser()
_held = None


def reclaimable_gb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(out.split("page size of")[1].split()[0])
    pages = {line.split(":")[0].strip(): int(line.split(":")[1].strip().rstrip(".")) for line in out.splitlines()[1:] if ":" in line and line.split(":")[1].strip().rstrip(".").isdigit()}
    return (pages.get("Pages free", 0) + pages.get("Pages inactive", 0) + pages.get("Pages speculative", 0) + pages.get("Pages purgeable", 0)) * page / 2**30


def acquire(job: str, need_gb: float, reserve_gb: float = 48.0, cache_gb: float = 4.0) -> None:
    """Refuse to start rather than put the host at risk. need_gb = expected peak of THIS job."""
    global _held
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = open(LOCK, "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        raise SystemExit(f"studio_guard: another Drift model process holds the lock ({handle.read().strip() or 'unknown'}); refusing to load a second model")
    available = reclaimable_gb()
    if available < need_gb + reserve_gb:
        raise SystemExit(f"studio_guard: {job} needs ~{need_gb:.0f} GB plus a {reserve_gb:.0f} GB reserve, but only {available:.0f} GB is reclaimable; not starting")
    handle.seek(0); handle.truncate(); handle.write(f"{job} pid {os.getpid()} since {time.strftime('%Y-%m-%d %H:%M:%S')} need {need_gb:.0f} GB\n"); handle.flush()
    _held = handle                                                 # the lock lives as long as the process
    import mlx.core as mx
    mx.set_memory_limit(int((need_gb + 8) * 2**30))                # exceed it and MLX raises; the host is not starved
    mx.set_cache_limit(int(cache_gb * 2**30))                      # freed buffers are returned instead of piling up across shapes
    print(f"studio_guard: {job} admitted ({available:.0f} GB reclaimable, limit {need_gb + 8:.0f} GB, cache {cache_gb:.0f} GB)", file=sys.stderr, flush=True)


def idle_exit(minutes: float = 15.0):
    """Returns touch(); the process exits if touch() is not called for `minutes` (orphaned worker after a dropped link)."""
    state = {"last": time.time()}

    def watch():
        while True:
            time.sleep(20)
            if time.time() - state["last"] > minutes * 60:
                print("studio_guard: idle too long, exiting and releasing the model", file=sys.stderr, flush=True)
                os._exit(0)

    threading.Thread(target=watch, daemon=True).start()

    def touch():
        state["last"] = time.time()
    return touch
