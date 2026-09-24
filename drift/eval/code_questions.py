"""Questions about Python source whose answers sit literally in it, for drop-in tests and answer-level training.

Reads source with ast, never runs it. A context is files under a header with their path. Kinds:
  default   the default value of a function's parameter, kept only when it is a distinctive number or string;
  constant  the value a module assigns to an upper-case name;
  raises    which function fails with a given message, kept only when the message is unique in the context;
  signature write a function's def line, for functions with two or more parameters and a name unique in the context;
            the answer is the def line itself, scored by parsing (drift.eval.signatures), not by containment;
  recall    write out a function exactly as the file has it, for functions of four or more lines with a name unique
            in the context; the answer is the opening of its def line. It is for training: the teacher's written-out
            function is the target, so every token of the body has to come through the memory.
"""
from __future__ import annotations
import ast
import re
from drift.eval.signatures import def_line

KINDS = ("default", "constant", "raises", "signature")                  # the drop-in gates' kinds; recall is asked for by name
ALL_KINDS = KINDS + ("recall",)
_FILE = re.compile(r"### (?P<path>[^\n]+)\n```python\n(?P<source>.*?)\n```\n", re.S)


def context_text(files: list[tuple[str, str]]) -> str:
    """(path, source) pairs -> the context, each file under a header with its path."""
    return "\n".join(f"### {relative}\n```python\n{source.rstrip()}\n```\n" for relative, source in files)


def split_context(context: str) -> list[tuple[str, str]]:
    """The context -> its (path, source) pairs; the inverse of context_text for sources without a fence line."""
    files = [(m["path"], m["source"]) for m in _FILE.finditer(context)]
    if context_text(files) != context:
        raise ValueError("the context is not files under path headers")
    return files


def distinctive(value) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return abs(value) >= 3 and value not in (10, 100)
    return isinstance(value, str) and 3 <= len(value) <= 60 and "\n" not in value


def message(call: ast.Call) -> str | None:
    """The literal message of raise X("...") or require(cond, "...")."""
    texts = [a.value for a in call.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    return texts[-1] if texts and 12 <= len(texts[-1]) <= 90 else None


def questions(files: list[tuple[str, str]], kinds=KINDS) -> list[dict]:
    """Unique questions [{kind, file, question, answer}] about the files, in the order the kinds are found."""
    unknown = set(kinds) - set(ALL_KINDS)
    if unknown:
        raise ValueError(f"unknown kinds: {sorted(unknown)}")
    found, messages, defined = [], {}, {}
    for relative, source in files:
        tree = ast.parse(source)
        for node in tree.body:                                         # module constants
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name.isupper() and isinstance(node.value, ast.Constant) and distinctive(node.value.value):
                    found.append({"kind": "constant", "file": relative, "question": f"In {relative}, what value is assigned to {name}?",
                                  "answer": str(node.value.value)})
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            defined.setdefault(node.name, []).append((relative, node))
            positional = node.args.posonlyargs + node.args.args
            pairs = list(zip(positional[len(positional) - len(node.args.defaults):], node.args.defaults))
            pairs += [(a, d) for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults) if d is not None]
            for arg, default in pairs:
                if isinstance(default, ast.Constant) and distinctive(default.value):
                    found.append({"kind": "default", "file": relative, "answer": str(default.value),
                                  "question": f"In {relative}, what is the default value of the {arg.arg} parameter of {node.name}?"})
            for inner in ast.walk(node):
                call = inner.exc if isinstance(inner, ast.Raise) and isinstance(inner.exc, ast.Call) else inner if isinstance(inner, ast.Call) and getattr(inner.func, "id", "") == "require" else None
                text = message(call) if call is not None else None
                if text:
                    messages.setdefault(text, set()).add((relative, node.name))
    for text, owners in messages.items():
        if len(owners) == 1:
            relative, function = next(iter(owners))
            found.append({"kind": "raises", "file": relative, "answer": function,
                          "question": f'Which function in {relative} fails with the message "{text}"? Give the function name.'})
    for name, places in defined.items():
        relative, node = places[0]
        arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        if len(places) == 1 and len([a for a in arguments if a.arg not in ("self", "cls")]) >= 2:
            found.append({"kind": "signature", "file": relative, "answer": def_line(node),
                          "question": f"Write the full def line of {name} in {relative}, with every parameter, annotation and default value, exactly as the file has it."})
    for name, places in defined.items():
        relative, node = places[0]
        if len(places) == 1 and node.end_lineno - node.lineno >= 3:
            opening = f"{'async def' if isinstance(node, ast.AsyncFunctionDef) else 'def'} {name}("
            found.append({"kind": "recall", "file": relative, "answer": opening,
                          "question": f"Write out {name} from {relative} exactly as the file has it, from its def line to the end of its body."})
    seen, unique = set(), []
    for q in found:
        if q["kind"] in kinds and q["question"] not in seen:
            seen.add(q["question"])
            unique.append(q)
    return unique
