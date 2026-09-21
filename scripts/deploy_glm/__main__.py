"""Prepare locally or inspect both Spark containers; never deploy or restart."""
import argparse
import json
import subprocess
from pathlib import Path

from .bundle import load_manifest, pin_sources, prepare, validate_sources
from .check import check_targets, refresh_rollback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=["check", "prepare", "check-staged", "pin"], default="check")
    parser.add_argument("--manifest", type=Path, default=Path("configs/glm_connector_deployment.json"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidate")
    parser.add_argument("--refresh-rollback", action="store_true")
    args = parser.parse_args()
    if args.refresh_rollback and args.mode != "pin":
        parser.error("only pin accepts --refresh-rollback")
    manifest = load_manifest(args.manifest)
    if args.mode == "pin":
        if not args.output or args.candidate:
            parser.error("pin requires --output and accepts no --candidate")
        commit = subprocess.run(["git", "-C", str(args.root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=10).stdout.strip()
        pinned = pin_sources(args.root, manifest, commit)
        if args.refresh_rollback:
            pinned = refresh_rollback(pinned)
        with args.output.open("x") as handle:
            handle.write(json.dumps(pinned, indent=2) + "\n")
        print(json.dumps({"output": str(args.output), "promotion_performed": False}))
        return 0
    validate_sources(args.root, manifest)
    if args.mode == "prepare":
        if not args.output or args.candidate:
            parser.error("prepare requires --output and accepts no --candidate")
        result = prepare(args.root, manifest, args.output)
    else:
        if args.output or (args.mode == "check-staged") != bool(args.candidate):
            parser.error("only check-staged accepts and requires --candidate")
        result = check_targets(manifest, candidate=args.candidate)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if result.get("verdict") == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
