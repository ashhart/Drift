"""Inspect public appointment preregistration metadata without opening evidence."""

import argparse
import json
from pathlib import Path

from drift.eval.readiness import inspect_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    try:
        with args.plan.open("rb") as source:
            result = inspect_bytes(source.read(1_048_577))
    except (OSError, ValueError, UnicodeError, RecursionError):
        result = {"status": "BLOCKED", "scope": "plan_completeness_only",
                  "launch_authorized": False, "publication_authorized": False,
                  "blockers": ["plan is unreadable, malformed or too large"]}
    print(json.dumps(result, indent=2))
    return 2 if result["status"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
