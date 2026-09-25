"""Score a model's answer that should contain a function's def line against the def line in the source.

The answer may wrap the line in a code block or prose; the first line starting with "def " or "async def " is taken.
Two levels: `callable` compares what a caller needs (parameter names and kinds in order, and default values), `exact`
also compares annotations and the return annotation. Comparison is on parsed syntax, so spacing and quoting do not
matter. Nothing is executed.
"""
from __future__ import annotations
import ast
import re

_DEF = re.compile(r"^\s*(?:async\s+)?def\s+\w+\s*\(")


def def_line(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """The def line of a function as source, without decorators or body."""
    kind = ast.AsyncFunctionDef if isinstance(node, ast.AsyncFunctionDef) else ast.FunctionDef
    bare = kind(name=node.name, args=node.args, body=[ast.Pass()], decorator_list=[], returns=node.returns)
    if "type_params" in kind._fields:                                  # Python 3.12 and later
        bare.type_params = list(getattr(node, "type_params", []))
    return ast.unparse(ast.fix_missing_locations(bare)).split("\n", 1)[0]


def parse(line: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    try:
        tree = ast.parse(line.strip() + "\n    pass\n")
    except SyntaxError:
        return None
    node = tree.body[0] if tree.body else None
    return node if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None


def found(answer: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """The first def line in an answer that parses, joining continuation lines until the colon."""
    lines = answer.splitlines()
    for i, line in enumerate(lines):
        if not _DEF.match(line):
            continue
        text = line
        for more in lines[i + 1:i + 200]:                              # a header spread over many lines, one parameter each
            if parse(text) is not None:
                break
            text += " " + more.strip()
        node = parse(text)
        if node is not None:
            return node
    return None


def _parameters(node, annotations: bool) -> list:
    args = node.args
    code = lambda value: None if value is None else ast.dump(value, annotate_fields=False)
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    out = [("positional", a.arg, code(d), code(a.annotation) if annotations else None) for a, d in zip(positional, defaults)]
    if args.vararg:
        out.append(("varargs", args.vararg.arg, None, code(args.vararg.annotation) if annotations else None))
    out += [("keyword", a.arg, code(d), code(a.annotation) if annotations else None) for a, d in zip(args.kwonlyargs, args.kw_defaults)]
    if args.kwarg:
        out.append(("varkw", args.kwarg.arg, None, code(args.kwarg.annotation) if annotations else None))
    return out


def score(answer: str, expected: str) -> dict:
    """{'found': bool, 'callable': bool, 'exact': bool} for an answer against the expected def line."""
    truth, got = parse(expected), found(answer)
    if truth is None:
        raise ValueError("the expected def line does not parse")
    if got is None:
        return {"found": False, "callable": False, "exact": False}
    same_name = got.name == truth.name
    callable_ = same_name and _parameters(got, False) == _parameters(truth, False)
    exact = callable_ and _parameters(got, True) == _parameters(truth, True) and ast.dump(got.returns or ast.Constant(None)) == ast.dump(truth.returns or ast.Constant(None))
    return {"found": True, "callable": callable_, "exact": exact}
