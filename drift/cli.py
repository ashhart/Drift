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


EPILOG = """exit codes:
  0  PASSED     the requested operation succeeded
  1  FAILED     the operation ran and did not pass
  2  BLOCKED    created, but prerequisites are missing (they are listed in the output)
  3  INVALID    the request or the supplied evidence is not usable

examples:
  drift adapter list
  drift adapter scaffold my_model/torch --model-type my_model --output ./drift-adapter-my-model
  drift tap status ./local/taps/my-pair

Creating a tap project needs a checkpoint, adapter id, host and runtime lock for each
side; see docs/guides/TAP_CLI.md. Without qualification reports and translators the project is
written as BLOCKED and tells you which gates are missing. This command never loads a
backbone model, trains a translator or starts a remote service."""


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="drift",
        description="Inventory local checkpoints, check the evidence needed to link them, and scaffold adapters.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = root.add_subparsers(dest="command", required=True, metavar="{tap,adapter}")
    tap = commands.add_parser("tap", help="create and inspect two-model tap projects")
    tap_commands = tap.add_subparsers(dest="tap_command", required=True, metavar="{create,status}")
    create = tap_commands.add_parser("create", help="hash two checkpoints and record which gates they pass")
    create.add_argument("name", help="project name recorded in the inventory")
    member_arguments(create, "source")
    member_arguments(create, "target")
    create.add_argument("--output", type=Path, required=True, help="project directory to create; never overwritten")
    status = tap_commands.add_parser("status", help="rehash a project and recompute its verdict")
    status.add_argument("path", type=Path, help="an existing tap project directory")
    adapter = commands.add_parser("adapter", help="list installed adapters or scaffold a new one")
    adapter_commands = adapter.add_subparsers(dest="adapter_command", required=True, metavar="{list,scaffold}")
    adapter_commands.add_parser("list", help="show built-in and installed adapter loaders")
    scaffold = adapter_commands.add_parser("scaffold", help="create a separate adapter package with a blocked loader")
    scaffold.add_argument("adapter_id", help="for example my_model/torch")
    scaffold.add_argument("--model-type", required=True, help="the checkpoint config model_type this adapter admits")
    scaffold.add_argument("--output", type=Path, required=True, help="package directory to create")
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
