"""Questions about a window of Python source cut from longer files, for answer-level training. Never runs the code.

Windows start and end mid-file, so the whole text rarely parses. Functions are found line by line instead: a def line,
its header up to the colon (parsed on its own), and a body of more-indented lines. A function whose body runs to the
end of the window is incomplete. File headers ("### path") name the file for the lines below them. Kinds, each
checked by containment of its answer in the teacher's answer:
  signature  write a function's def line, for names defined once in the window with two or more parameters;
  recall     write out a complete function of four or more body lines, name defined once in the window;
  raises     which function fails with a message, for messages that appear once in the window;
  continue   quote the line that follows a distinctive line, both at least 20 characters;
  block      quote the eight lines that follow a distinctive line, at least 160 characters between them, all
             inside the window; the answer to contain is their first line;
  table      in a Markdown table, what one row says under another column, for rows whose first cell is unique in
             the table and cells of four or more characters; the answer is the cell's text.
"""
from __future__ import annotations
import ast
import re
import textwrap
from drift.eval.signatures import def_line

KINDS = ("signature", "recall", "raises", "continue", "block", "table")
BLOCK_LINES = 8
_DEF = re.compile(r"^(?P<indent>[ \t]*)(?:async[ \t]+)?def[ \t]+(?P<name>\w+)[ \t]*\(")
_HEADER = re.compile(r"^### (?P<path>\S+)$")
_MESSAGE = re.compile(r"""(?:raise\s+[\w.]+\(|require\(.*?,\s*)(?P<quote>["'])(?P<text>[^"'\\{}]{12,90})(?P=quote)""")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def functions(text: str) -> list[dict]:
    """Each function whose header lies whole in the window: name, file path or None, def line, body line range, and
    whether its body ends inside the window."""
    lines, found, path = text.split("\n"), [], None
    paths = []
    for line in lines:
        match = _HEADER.match(line)
        path = match["path"] if match else path
        paths.append(path)
    for i, line in enumerate(lines):
        match = _DEF.match(line)
        if not match:
            continue
        header_end = None
        for j in range(i, min(i + 12, len(lines))):
            code = lines[j].split("#")[0].rstrip()
            if code.endswith(":") and _balanced(" ".join(lines[i:j + 1])):
                header_end = j
                break
        if header_end is None:
            continue
        try:
            node = ast.parse(textwrap.dedent("\n".join(lines[i:header_end + 1])) + "\n    pass").body[0]
        except SyntaxError:
            continue
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        indent, end, complete = _indent(line), header_end + 1, False
        while end < len(lines):
            if lines[end].strip() and _indent(lines[end]) <= indent:
                complete = True
                break
            end += 1
        while end > header_end + 1 and not lines[end - 1].strip():
            end -= 1                                                    # trailing blank lines are not the body
        found.append({"name": match["name"], "path": paths[i], "def_line": def_line(node), "node": node,
                      "start": i, "body": (header_end + 1, end), "complete": complete})
    return found


def _balanced(text: str) -> bool:
    depth = 0
    for ch in text:
        depth += ch in "([{"
        depth -= ch in ")]}"
    return depth == 0


def questions(text: str, kinds=KINDS) -> list[dict]:
    """Questions [{kind, question, answer}] about one window; answers are what the teacher's answer must contain."""
    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise ValueError(f"unknown kinds: {sorted(unknown)}")
    lines, found = text.split("\n"), []
    funcs = functions(text)
    counts = {}
    for f in funcs:
        counts[f["name"]] = counts.get(f["name"], 0) + 1
    where = lambda f: f" in {f['path']}" if f["path"] else " in the project files"
    for f in funcs:
        if counts[f["name"]] != 1:
            continue
        arguments = f["node"].args.posonlyargs + f["node"].args.args + f["node"].args.kwonlyargs
        if "signature" in kinds and len([a for a in arguments if a.arg not in ("self", "cls")]) >= 2:
            found.append({"kind": "signature", "answer": f"def {f['name']}(",
                          "question": f"Write the full def line of {f['name']}{where(f)}, with every parameter, annotation and default value, exactly as the file has it."})
        if "recall" in kinds and f["complete"] and f["body"][1] - f["body"][0] >= 4:
            found.append({"kind": "recall", "answer": f"def {f['name']}(",
                          "question": f"Write out {f['name']}{where(f)} exactly as the file has it, from its def line to the end of its body."})
    if "raises" in kinds:
        owners = {}
        for f in funcs:
            for line in lines[f["body"][0]:f["body"][1]]:
                for match in _MESSAGE.finditer(line):
                    owners.setdefault(match["text"], []).append(f)
        occurrences = {}
        for line in lines:
            for match in _MESSAGE.finditer(line):
                occurrences[match["text"]] = occurrences.get(match["text"], 0) + 1
        for message, fs in owners.items():
            if occurrences[message] == 1 and len(fs) == 1 and counts[fs[0]["name"]] == 1:
                found.append({"kind": "raises", "answer": fs[0]["name"],
                              "question": f'Which function{where(fs[0])} fails with the message "{message}"? Give the function name.'})
    if "continue" in kinds:
        stripped = [line.strip() for line in lines]
        seen = {}
        for s in stripped:
            seen[s] = seen.get(s, 0) + 1
        for i in range(len(lines) - 1):
            here, after = stripped[i], stripped[i + 1]
            if len(here) >= 20 and len(after) >= 20 and seen[here] == 1 and not here.startswith(("###", "```")) and not after.startswith(("###", "```")):
                found.append({"kind": "continue", "answer": after,
                              "question": f"In the project files, which line comes right after this one? Quote it exactly.\n{here}"})
    if "block" in kinds:
        stripped = [line.strip() for line in lines]
        seen = {}
        for s in stripped:
            seen[s] = seen.get(s, 0) + 1
        for i in range(len(lines) - BLOCK_LINES - 1):                   # all eight lines lie in the window; a final newline adds an empty line
            here, block = stripped[i], lines[i + 1:i + 1 + BLOCK_LINES]
            if (len(here) >= 20 and seen[here] == 1 and not here.startswith(("###", "```")) and stripped[i + 1]
                    and sum(len(b.strip()) for b in block) >= 160 and not any(b.strip().startswith(("###", "```")) for b in block)):
                found.append({"kind": "block", "answer": stripped[i + 1],
                              "question": f"In the project files, quote the {BLOCK_LINES} lines that come right after this one, exactly as they are.\n{here}"})
    if "table" in kinds:
        found += _table_questions(lines)
    return found


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _table_questions(lines: list[str]) -> list[dict]:
    """What a Markdown table's row says under another column, one question per row and column."""
    found, i = [], 0
    while i < len(lines) - 2:
        if not (lines[i].strip().startswith("|") and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1])):
            i += 1
            continue
        header, j, rows = _cells(lines[i]), i + 2, []
        while j < len(lines) and lines[j].strip().startswith("|"):
            rows.append(_cells(lines[j]))
            j += 1
        keys = [row[0] for row in rows]
        for row in rows:
            if len(row) != len(header) or not row[0] or keys.count(row[0]) != 1:
                continue
            for column in range(1, len(header)):
                if len(row[column]) >= 4 and header[column]:
                    found.append({"kind": "table", "answer": row[column],
                                  "question": f"In the project files, a table with the columns {', '.join(header)} has a row for {row[0]}. "
                                              f"What does that row say under {header[column]}? Quote it exactly."})
        i = j
    return found
