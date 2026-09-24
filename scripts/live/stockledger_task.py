"""A drop-in coding task on the Stockledger benchmark: implement its event reader from the specification alone.

The task text gives only the interface. The field rules (formats, ranges, rejected encodings, line numbers) are in the
project's SPEC.md, which only the resident model has read; the joining model gets them from Drift memory, from the text,
or not at all. Hidden checks, derived from the specification's event-schema section and never shown to either model,
grade the returned module in an isolated Python process with a bare environment and a timeout. Nothing here runs the
benchmark repository's own code or changes it.
"""
from __future__ import annotations
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TASK = ("Using the Stockledger project's specification from our shared memory, write the complete module "
        "stockledger/validation.py. It must define read_events(stream): stream is a binary file object holding JSON Lines "
        "input. Return the events as a list of dicts in input order, each holding the event's fields with the values the input "
        "gave. On the first line the specification rejects, raise "
        "ValueError whose message contains that line's 1-based line number. Standard library only. Reply with the module "
        "in one python code block.")
CONTEXT_FILES = ("SPEC.md", "stockledger/validation.py", "stockledger/cli.py")

GOOD = {"event_id": "public-1", "occurred_at": "2026-01-02T10:00:00Z", "warehouse": "north", "sku": "BOLT-M8", "delta": 12}


def line(**changes) -> str:
    event = {**GOOD, **changes}
    return json.dumps({k: v for k, v in event.items() if v is not ...})


# name, input bytes, and either the events expected or the 1-based line number the error must name
CHECKS = [
    ("one valid event", (line() + "\n").encode(), [GOOD]),
    ("no final newline", line().encode(), [GOOD]),
    ("blank lines are ignored", (" \t\r\n" + line() + "\n\n").encode(), [GOOD]),
    ("field order does not matter", (json.dumps(dict(reversed(list(GOOD.items())))) + "\n").encode(), [GOOD]),
    ("delta zero and bounds", "\n".join(line(event_id=f"e{i}", delta=d) for i, d in enumerate((0, -1000000000, 1000000000))).encode(),
     [{**GOOD, "event_id": f"e{i}", "delta": d} for i, d in enumerate((0, -1000000000, 1000000000))]),
    ("longest identifiers", line(event_id="a" * 64, warehouse="w" * 32, sku="S" * 64).encode(), [{**GOOD, "event_id": "a" * 64, "warehouse": "w" * 32, "sku": "S" * 64}]),
    ("allowed identifier characters", line(event_id="A.b_c-9", warehouse="north-2", sku="BOLT_M8-2").encode(),
     [{**GOOD, "event_id": "A.b_c-9", "warehouse": "north-2", "sku": "BOLT_M8-2"}]),
    ("first and last years", "\n".join((line(event_id="a", occurred_at="0001-01-01T00:00:00Z"), line(event_id="b", occurred_at="9999-12-31T23:59:59Z"))).encode(),
     [{**GOOD, "event_id": "a", "occurred_at": "0001-01-01T00:00:00Z"}, {**GOOD, "event_id": "b", "occurred_at": "9999-12-31T23:59:59Z"}]),
    ("missing field", line(sku=...).encode(), 1),
    ("unknown field", json.dumps({**GOOD, "note": "x"}).encode(), 1),
    ("duplicate key", line()[:-1].encode() + b', "delta": 12}', 1),
    ("null value", line(sku=None).encode(), 1),
    ("malformed JSON", b'{"event_id": "x",', 1),
    ("not an object", b"[1, 2]", 1),
    ("invalid UTF-8", line().encode().replace(b"north", b"n\xffrth"), 1),
    ("UTF-8 byte order mark", b"\xef\xbb\xbf" + line().encode(), 1),
    ("empty event_id", line(event_id="").encode(), 1),
    ("event_id too long", line(event_id="a" * 65).encode(), 1),
    ("event_id with a space", line(event_id="a b").encode(), 1),
    ("timestamp without Z", line(occurred_at="2026-01-02T10:00:00").encode(), 1),
    ("timestamp with fractions", line(occurred_at="2026-01-02T10:00:00.5Z").encode(), 1),
    ("impossible date", line(occurred_at="2026-02-30T10:00:00Z").encode(), 1),
    ("leap second", line(occurred_at="2026-06-30T23:59:60Z").encode(), 1),
    ("uppercase warehouse", line(warehouse="North").encode(), 1),
    ("warehouse too long", line(warehouse="w" * 33).encode(), 1),
    ("lowercase sku", line(sku="bolt-m8").encode(), 1),
    ("sku with a dot", line(sku="BOLT.M8").encode(), 1),
    ("float delta", line(delta=1.0).encode(), 1),
    ("boolean delta", line(delta=True).encode(), 1),
    ("string delta", line(delta="5").encode(), 1),
    ("delta out of range", line(delta=1000000001).encode(), 1),
    ("error names the physical line", ("\n".join((line(event_id="a"), "", line(event_id="b", sku="bad")))).encode(), 3),
    ("a value is not trimmed", line(warehouse=" north").encode(), 1),
]

HARNESS = r'''
import importlib.util, io, json, re, sys
spec = importlib.util.spec_from_file_location("validation", sys.argv[1])
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
data = open(sys.argv[2], "rb").read()
try:
    events = module.read_events(io.BytesIO(data))
    print(json.dumps({"events": [dict(e) for e in events]}))
except Exception as error:
    print(json.dumps({"error": type(error).__name__, "message": str(error)[:300]}))
'''


def code_block(text: str) -> str | None:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else None


def grade(answer: str, timeout: float = 10) -> dict:
    """Each hidden check run in its own isolated process: {'passed', 'total', 'results': {name: bool}}."""
    code, results = code_block(answer), {}
    with tempfile.TemporaryDirectory() as folder:
        module, harness = Path(folder, "validation.py"), Path(folder, "harness.py")
        module.write_text(code or "raise SystemExit('no code block')\n")
        harness.write_text(HARNESS)
        for name, data, expected in CHECKS:
            source = Path(folder, "input.jsonl")
            source.write_bytes(data)
            try:
                run = subprocess.run([sys.executable, "-I", str(harness), str(module), str(source)], cwd=folder, capture_output=True,
                                     text=True, timeout=timeout, env={"PATH": "/usr/bin:/bin"})
                outcome = json.loads(run.stdout.strip().splitlines()[-1]) if run.returncode == 0 and run.stdout.strip() else {}
            except (subprocess.TimeoutExpired, ValueError, IndexError):
                outcome = {}
            if isinstance(expected, int):                               # the message must name the line, not merely contain the digit
                message = outcome.get("message", "")
                named = re.search(rf"(?i)\bline\W{{0,3}}{expected}\b", message) or re.match(rf"\s*{expected}\s*:", message)
                results[name] = outcome.get("error") == "ValueError" and named is not None
            else:
                results[name] = outcome.get("events") == expected
    return {"passed": sum(results.values()), "total": len(results), "code": code is not None, "results": results}
