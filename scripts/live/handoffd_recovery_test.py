"""Does MCDMA recover on its own? Breaks pulls on purpose and checks that the next pull lands byte for byte.

Runs from the orchestrating machine over SSH. On each Spark it writes a random blob into the handoff root and records
its SHA-256; on the Studio it drives handoffd through drift/serving/handoffd_client.py and hashes what each pull wrote
into shared memory. Two ways to break a pull, neither restarting a daemon:
  orphan  the Studio forgets its control connection without closing it (TEST_ORPHAN), as a connection that died
          silently leaves the Spark's end open; tonight's failure mode.
  stall   the Spark's daemon is paused (SIGSTOP) before a pull, so the pull must fail within the daemon's bounds
          rather than hang, then resumed (SIGCONT).
After each break the next pull must succeed and match the blob's SHA-256. Prints one JSON report.

  python3 scripts/live/handoffd_recovery_test.py --studio STUDIO_SSH --spark spark-a=SPARK_SSH --repeats 3
"""
from __future__ import annotations
import argparse
import json
import subprocess
import time

PULL = r'''
import hashlib, json, sys, time
sys.path.insert(0, sys.argv[1])
from drift.serving.handoffd_client import HandoffdClient, HandoffdError
client = HandoffdClient("/tmp/handoffd.sock", timeout_s=120)
peer, remote, action = sys.argv[2], sys.argv[3], sys.argv[4]
started = time.time()
try:
    if action == "orphan":
        client.test_orphan(peer)
    pulled = client.pull(peer, remote, offset=0, unlink=False)
    digest = hashlib.sha256(bytes(client.view(peer, 0, pulled.bytes))).hexdigest()
    print(json.dumps({"ok": True, "bytes": pulled.bytes, "gbit_s": round(pulled.gbit_s, 1), "sha256": digest, "seconds": round(time.time() - started, 3)}))
except HandoffdError as error:
    print(json.dumps({"ok": False, "error": str(error), "seconds": round(time.time() - started, 3)}))
'''


def ssh(host: str, command: str, stdin: str | None = None, timeout: float = 300) -> str:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, command], input=stdin, capture_output=True, text=True,
                          timeout=timeout, check=True).stdout.strip()


def pull(studio: str, repo: str, python: str, peer: str, remote: str, action: str = "pull") -> dict:
    return json.loads(ssh(studio, f"{python} - {repo} {peer} {remote} {action}", stdin=PULL).splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--studio", required=True, help="SSH name of the Studio running handoffd studio")
    parser.add_argument("--spark", action="append", required=True, help="PEER=SSH_NAME for each Spark to test")
    parser.add_argument("--repo", default="drift-frontier", help="the Drift checkout on the Studio, relative to home")
    parser.add_argument("--python", default="/usr/bin/python3")
    parser.add_argument("--megabytes", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--stall", action="store_true", help="also pause the Spark's daemon before a pull")
    args = parser.parse_args()
    report = {}
    for pair in args.spark:
        peer, host = pair.split("=", 1)
        remote = "/dev/shm/glm53-handoff/recovery-test/blob.bin"
        expected = ssh(host, f"mkdir -p {remote.rsplit('/', 1)[0]} && head -c {args.megabytes * 1048576} /dev/urandom > {remote} && sha256sum {remote} | cut -c1-64")
        runs = [{"step": "baseline", **pull(args.studio, args.repo, args.python, peer, remote)}]
        for i in range(args.repeats):
            runs.append({"step": f"orphan then pull {i + 1}", **pull(args.studio, args.repo, args.python, peer, remote, "orphan")})
        if args.stall:
            pid = ssh(host, "pgrep -f '^./handoffd spark' | head -1")
            ssh(host, f"kill -STOP {pid}")
            try:
                runs.append({"step": "pull while the Spark's daemon is paused", **pull(args.studio, args.repo, args.python, peer, remote)})
            finally:
                ssh(host, f"kill -CONT {pid}")
            time.sleep(1)
            runs.append({"step": "pull after it resumes", **pull(args.studio, args.repo, args.python, peer, remote)})
        for run in runs:
            run["matches"] = run.get("sha256") == expected
        ssh(host, f"rm -f {remote} && rmdir {remote.rsplit('/', 1)[0]}")
        report[peer] = {"blob_sha256": expected, "runs": runs}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
