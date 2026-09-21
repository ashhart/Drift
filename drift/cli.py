from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from drift.adapters.catalog import adapter_records
from drift.taps.project import create_tap
from drift.taps.scaffold import scaffold_adapter
from drift.taps.status import tap_status


EXIT_CODES = {"PASSED": 0, "FAILED": 1, "BLOCKED": 2, "INVALID": 3}


def member_arguments(parser: argparse.ArgumentParser, role: str) -> None:
    parser.add_argument(f"--{role}-checkpoint", type=Path, required=True)
    parser.add_argument(f"--{role}-adapter", required=True)
    parser.add_argument(f"--{role}-host", required=True)
    parser.add_argument(f"--{role}-runtime-lock", type=Path, required=True)
    parser.add_argument(f"--{role}-qualification", type=Path)
    parser.add_argument(f"--{role}-translator", type=Path)
    parser.add_argument(f"--{role}-quantization", default="none")


def member_spec(args: argparse.Namespace, role: str) -> dict:
    return {
        "checkpoint": getattr(args, f"{role}_checkpoint"),
        "adapter": getattr(args, f"{role}_adapter"),
        "host": getattr(args, f"{role}_host"),
        "runtime_lock": getattr(args, f"{role}_runtime_lock"),
        "qualification": getattr(args, f"{role}_qualification"),
        "translator": getattr(args, f"{role}_translator"),
        "quantization": getattr(args, f"{role}_quantization"),
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="drift")
    commands = root.add_subparsers(dest="command", required=True)
    tap = commands.add_parser("tap")
    tap_commands = tap.add_subparsers(dest="tap_command", required=True)
    create = tap_commands.add_parser("create")
    create.add_argument("name")
    member_arguments(create, "source")
    member_arguments(create, "target")
    create.add_argument("--output", type=Path, required=True)
    status = tap_commands.add_parser("status")
    status.add_argument("path", type=Path)
    adapter = commands.add_parser("adapter")
    adapter_commands = adapter.add_subparsers(dest="adapter_command", required=True)
    adapter_commands.add_parser("list")
    scaffold = adapter_commands.add_parser("scaffold")
    scaffold.add_argument("adapter_id")
    scaffold.add_argument("--model-type", required=True)
    scaffold.add_argument("--output", type=Path, required=True)
    return root


def dispatch(args: argparse.Namespace) -> dict | list[dict]:
    if args.command == "tap" and args.tap_command == "create":
        return create_tap(args.name, member_spec(args, "source"), member_spec(args, "target"), args.output)
    if args.command == "tap" and args.tap_command == "status":
        return tap_status(args.path)
    if args.command == "adapter" and args.adapter_command == "list":
        return [vars(record) for record in adapter_records()]
    if args.command == "adapter" and args.adapter_command == "scaffold":
        return scaffold_adapter(args.adapter_id, args.model_type, args.output)
    raise ValueError("unknown command")


def main() -> None:
    try:
        result = dispatch(parser().parse_args())
    except (OSError, ValueError) as error:
        result = {"status": "INVALID", "error": str(error)}
        print(json.dumps(result, indent=2), file=sys.stderr)
        raise SystemExit(EXIT_CODES["INVALID"])
    print(json.dumps(result, indent=2))
    status = result.get("status", "PASSED") if isinstance(result, dict) else "PASSED"
    raise SystemExit(EXIT_CODES[status])


if __name__ == "__main__":
    main()
