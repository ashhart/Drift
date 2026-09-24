"""A held-out drop-in coding task: write iniconfig's INI parser from a specification of its behaviour.

iniconfig (2.3.0) is a small public package that no translator, question set or checkpoint choice here has seen. The
context holds SPEC.md, which states the parser's rules, and the package's own exceptions.py, __init__.py and a stub of
_parse.py; the parser's real source is not in it. The joining model gets the context from Drift memory, from text, or
not at all, and writes iniconfig/_parse.py. Hidden checks, never shown to either model, run the returned module on INI
inputs in isolated Python processes with a bare environment and a timeout, and compare what it returns or raises with
what the real parser does; the expected results are recorded below.

  python3 scripts/live/iniparse_task.py --root .venv/lib/python3.12/site-packages --out iniparse_items.json
"""
from __future__ import annotations
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SPEC = """# iniconfig: parsing INI data

`parse_ini_data(path, data, *, strip_inline_comments, strip_section_whitespace=False)` parses INI text and returns
`(sections, sources)`. `sections` maps each section name to a dict of its names and values, both in file order.
`sources` maps `(section, None)` to the line index of the section's header and `(section, name)` to the line index of
the line that defined the name. Lines are `data.splitlines(True)`; line indexes count from 0.

## Kinds of line

1. Blank or comment. A line whose first character that is not whitespace is `#` or `;`, or a line that is empty once
   trailing whitespace is removed. It is skipped.
2. Section header. Once trailing whitespace is removed, the line starts with `[`. Cut it at the first `#`, remove
   trailing whitespace, then cut at the first `;` and remove trailing whitespace again. If the result ends with `]`, the
   section name is everything between its first character and that last `]`, kept exactly, spaces included; with
   `strip_section_whitespace` true it is stripped. If the result does not end with `]`, the line is a continuation whose
   text is the whole original line stripped.
3. Name and value. The line starts with a character that is neither whitespace nor `[`. Split it at the first `=`.
   If it has no `=`, or the part before the first `=` contains a `:`, split it at the first `:` instead. If neither
   splits it, that is an error. The name is the part before, stripped; the value is the part after, stripped. With
   `strip_inline_comments` true the value is cut at the first `#` with trailing whitespace removed, then at the first
   `;` with trailing whitespace removed.
4. Continuation. Any other line that starts with whitespace. Its text is the line stripped, and with
   `strip_inline_comments` true it is cut at `#` then `;` the same way.

## Building the result

- A name and value belong to the section most recently opened.
- A continuation adds to the value of the most recent name: if that value is empty, the continuation's text replaces
  it; otherwise the value becomes the value, a newline, and the text.

## Errors

Every error is `ParseError(path, lineno, msg)` from `iniconfig.exceptions`, where `lineno` is the index of the line at
fault and `msg` is exactly one of these:

- `empty section name`: a section header whose name is empty.
- `unexpected value continuation`: a continuation with no name and value before it, at the start of the data or
  straight after a section header.
- `unexpected line: 'TEXT'`: a line that neither `=` nor `:` splits; TEXT is the line with trailing whitespace
  removed, written as Python's `repr` writes it.
- `no section header defined`: a name and value before any section header.
- `duplicate section 'NAME'`: a section opened a second time, NAME written as `repr` writes it.
- `duplicate name 'NAME'`: a name defined twice in one section.

The first three are found while reading the lines, over the whole data, before any of the last three is looked for.
The last three are then found in line order.
"""

STUB = '''"""INI parsing for iniconfig. Implements parse_ini_data as SPEC.md describes."""
from collections.abc import Mapping

from .exceptions import ParseError

COMMENTCHARS = "#;"


def parse_ini_data(
    path: str,
    data: str,
    *,
    strip_inline_comments: bool,
    strip_section_whitespace: bool = False,
) -> tuple[Mapping[str, Mapping[str, str]], Mapping[tuple[str, str | None], int]]:
    raise NotImplementedError("Implement INI parsing as SPEC.md describes")


def iscommentline(line: str) -> bool:
    raise NotImplementedError("Implement comment detection as SPEC.md describes")
'''

TASK = ("Using the iniconfig project's specification and files from our shared memory, write the complete module "
        "iniconfig/_parse.py. It must define COMMENTCHARS, iscommentline(line) and parse_ini_data(path, data, *, "
        "strip_inline_comments, strip_section_whitespace=False) returning (sections, sources) exactly as SPEC.md states, "
        "and raise ParseError imported from .exceptions with the index of the line at fault and the exact message. "
        "Standard library only. Write the code directly, with no planning in comments. Reply with the module in one "
        "python code block.")
CONTEXT_FILES = ("iniconfig/exceptions.py", "iniconfig/__init__.py")

# name, INI data, keyword arguments; expected results come from iniconfig 2.3.0's own parse_ini_data (see EXPECTED)
CASES = [
    ("names and values", "[a]\nx = 1\ny=2\n", {}),
    ("colon separator", "[a]\nx: 1\n", {}),
    ("a colon before the first equals sign", "[a]\nx:y = 1\n", {}),
    ("an equals sign inside the value", "[a]\nurl = a=b\n", {}),
    ("a colon inside the value", "[a]\nx = http://h:1\n", {}),
    ("comment lines", "# c\n[a]\n; c\nx = 1\n  # indented comment\n", {}),
    ("blank lines", "\n[a]\n\n\nx = 1\n\n", {}),
    ("continuation lines", "[a]\nx = 1\n  2\n  3\n", {}),
    ("a continuation of an empty value", "[a]\nx =\n  first\n  second\n", {}),
    ("a tab continuation", "[a]\nx = 1\n\t2\n", {}),
    ("an inline comment is kept by default", "[a]\nx = 1 # note\n", {}),
    ("an inline comment is stripped when asked", "[a]\nx = 1 # note\n", {"strip_inline_comments": True}),
    ("a semicolon comment is stripped when asked", "[a]\nx = 1 ; note\n", {"strip_inline_comments": True}),
    ("a continuation's comment is stripped when asked", "[a]\nx = 1\n  2 # c\n", {"strip_inline_comments": True}),
    ("a header's trailing comment", "[a] # c\nx = 1\n", {}),
    ("a header keeps its spaces", "[ a ]\nx=1\n", {}),
    ("a header's spaces are stripped when asked", "[ a ]\nx=1\n", {"strip_section_whitespace": True}),
    ("a bracket inside a header", "[a]b]\nx=1\n", {}),
    ("line indexes in sources", "\n[a]\n\nx = 1\n[b]\ny = 2\n", {}),
    ("Windows line endings", "[a]\r\nx = 1\r\n", {}),
    ("empty data", "", {}),
    ("only comments", "# a\n; b\n", {}),
    ("names and values are stripped", "[a]\nx   =   1   \n", {}),
    ("an unclosed header continues the value before it", "[a]\nx = 1\n[b\n", {}),
    ("a value before any header", "x = 1\n", {}),
    ("a section opened twice", "[a]\n[a]\n", {}),
    ("a name defined twice", "[a]\nx=1\nx=2\n", {}),
    ("an empty section name", "[a]\nx=1\n[]\n", {}),
    ("a continuation at the start", "  x\n", {}),
    ("a continuation after a header", "[a]\n  x\n", {}),
    ("a line that nothing splits", "[a]\nnovalue\n", {}),
    ("reading errors come before section errors", "[a]\n[a]\nbad line\n", {}),
    ("the same name in two sections", "[a]\nx=1\n[b]\nx=2\n", {}),
]

HARNESS = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
case = json.load(open(sys.argv[2]))
try:
    from iniconfig._parse import parse_ini_data
    kwargs = {"strip_inline_comments": False, **case["kwargs"]}
    sections, sources = parse_ini_data("config.ini", case["data"], **kwargs)
    print(json.dumps({"sections": {s: dict(v) for s, v in sections.items()},
                      "sources": sorted(([s, n, i] for (s, n), i in sources.items()), key=lambda t: (t[0], t[1] or "", t[2]))}))
except Exception as error:
    print(json.dumps({"error": type(error).__name__, "lineno": getattr(error, "lineno", None), "msg": str(getattr(error, "msg", error))[:300]}))
'''


def outcome(parse_ini_data, data: str, kwargs: dict) -> dict:
    """What a parse_ini_data does with one case, in the harness's form."""
    try:
        sections, sources = parse_ini_data("config.ini", data, **{"strip_inline_comments": False, **kwargs})
        return {"sections": {s: dict(v) for s, v in sections.items()},
                "sources": sorted(([s, n, i] for (s, n), i in sources.items()), key=lambda t: (t[0], t[1] or "", t[2]))}   # a header's name is None
    except Exception as error:
        return {"error": type(error).__name__, "lineno": getattr(error, "lineno", None), "msg": str(getattr(error, "msg", error))[:300]}


EXPECTED_FILE = Path(__file__).with_name("iniparse_expected.json")


def code_block(text: str) -> str | None:
    """The longest code block of the answer; with thinking allowed, only what follows the thinking is the answer."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text.rsplit("</think>", 1)[-1], re.S)
    return max(blocks, key=len) if blocks else None


def grade(answer: str, exceptions_source: str, timeout: float = 10) -> dict:
    """Each hidden check in its own isolated process: {'passed', 'total', 'code', 'results': {name: bool}}."""
    expected = json.loads(EXPECTED_FILE.read_text())
    code, results = code_block(answer), {}
    with tempfile.TemporaryDirectory() as folder:
        package = Path(folder, "iniconfig")
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "exceptions.py").write_text(exceptions_source)
        (package / "_parse.py").write_text(code or "raise SystemExit('no code block')\n")
        harness = Path(folder, "harness.py")
        harness.write_text(HARNESS)
        for name, data, kwargs in CASES:
            case = Path(folder, "case.json")
            case.write_text(json.dumps({"data": data, "kwargs": kwargs}))
            try:
                run = subprocess.run([sys.executable, "-I", str(harness), folder, str(case)], cwd=folder, capture_output=True,
                                     text=True, timeout=timeout, env={"PATH": "/usr/bin:/bin"})
                got = json.loads(run.stdout.strip().splitlines()[-1]) if run.returncode == 0 and run.stdout.strip() else {}
            except (subprocess.TimeoutExpired, ValueError, IndexError):
                got = {}
            results[name] = got == expected[name]
    return {"passed": sum(results.values()), "total": len(results), "code": code is not None, "results": results}


def context(root: Path) -> str:
    """The task's context: the specification, the package's own exceptions and __init__, and the parser's stub."""
    parts = [f"### SPEC.md\n```markdown\n{SPEC.rstrip()}\n```\n"]
    parts += [f"### {name}\n```python\n{(root / name).read_text().rstrip()}\n```\n" for name in CONTEXT_FILES]
    parts.append(f"### iniconfig/_parse.py\n```python\n{STUB.rstrip()}\n```\n")
    return "\n".join(parts)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="the site-packages folder holding iniconfig 2.3.0")
    parser.add_argument("--out", type=Path, required=True, help="gate items for the task")
    parser.add_argument("--record-expected", action="store_true", help="record the real parser's results for the hidden checks")
    parser.add_argument("--think", action="store_true", help="let the joining model think before it answers, instead of the empty think block")
    args = parser.parse_args()
    if args.record_expected:
        sys.path.insert(0, str(args.root))
        from iniconfig._parse import parse_ini_data
        EXPECTED_FILE.write_text(json.dumps({name: outcome(parse_ini_data, data, kwargs) for name, data, kwargs in CASES}, indent=1) + "\n")
    from gate_items import HEAD, tail
    passage = context(args.root)
    args.out.write_text(json.dumps([{"id": "i1-iniparse", "kind": "coding", "passage": passage, "question": TASK, "answer": "",
                                     "head_text": HEAD, "tail_text": tail(TASK).split("<think>")[0] if args.think else tail(TASK)}], indent=1) + "\n")
    print(json.dumps({"passage_chars": len(passage), "checks": len(CASES)}))
