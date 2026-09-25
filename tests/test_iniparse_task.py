"""The held-out INI task grades by its hidden checks: the real parser passes all, the stub none, a near miss some."""
import glob
from pathlib import Path
import pytest
from scripts.live.iniparse_task import CASES, STUB, grade

ROOT = Path(sorted(glob.glob(".venv/lib/python3*/site-packages"))[0]) if glob.glob(".venv/lib/python3*/site-packages") else None
needs_iniconfig = pytest.mark.skipif(ROOT is None or not (ROOT / "iniconfig" / "_parse.py").exists(), reason="iniconfig is not installed here")


def answer(code: str) -> str:
    return f"Here is the module.\n```python\n{code}\n```\n"


@needs_iniconfig
def test_the_real_parser_passes_every_hidden_check_and_the_stub_none():
    exceptions = (ROOT / "iniconfig" / "exceptions.py").read_text()
    real = grade(answer((ROOT / "iniconfig" / "_parse.py").read_text()), exceptions)
    assert real["code"] and real["passed"] == real["total"] == len(CASES)
    stub = grade(answer(STUB), exceptions)
    assert stub["passed"] == 0 and grade("no code at all", exceptions)["code"] is False


@needs_iniconfig
def test_a_parser_without_the_colon_rule_fails_only_the_checks_that_need_it():
    exceptions = (ROOT / "iniconfig" / "exceptions.py").read_text()
    source = (ROOT / "iniconfig" / "_parse.py").read_text()
    broken = source.replace('            if ":" in name:\n                raise ValueError()\n', "")
    assert broken != source
    results = grade(answer(broken), exceptions)["results"]
    assert not results["a colon before the first equals sign"]
    assert sum(not ok for ok in results.values()) == 1


def test_only_the_answer_after_thinking_is_graded():
    from scripts.live.iniparse_task import code_block
    draft = "```python\n" + "x = 1\n" * 50 + "```"
    answer = "<think>" + draft + "</think>\n```python\nCOMMENTCHARS = '#;'\n```"
    assert code_block(answer) == "COMMENTCHARS = '#;'\n"
    assert code_block(draft) == "x = 1\n" * 50



# ParseError as iniconfig.exceptions defines it, so the next check needs no installed iniconfig
EXCEPTIONS = '''class ParseError(Exception):
    def __init__(self, path, lineno, msg):
        super().__init__(path, lineno, msg)
        self.path, self.lineno, self.msg = path, lineno, msg
'''

# A parser written from SPEC_TWO_PASS alone, rule by rule
TWO_PASS = '''from .exceptions import ParseError

COMMENTCHARS = "#;"


def iscommentline(line):
    return line.lstrip()[:1] in ("", "#", ";")


def _cut(text):
    return text.split("#", 1)[0].rstrip().split(";", 1)[0].rstrip()


def parse_ini_data(path, data, *, strip_inline_comments, strip_section_whitespace=False):
    records, section = [], None
    for lineno, line in enumerate(data.splitlines(True)):
        stripped = line.rstrip()
        if iscommentline(line):
            continue
        if stripped.startswith("["):
            head = _cut(stripped)
            if head.endswith("]"):
                name = head[1:-1].strip() if strip_section_whitespace else head[1:-1]
                if not name:
                    raise ParseError(path, lineno, "empty section name")
                section = name
                records.append([lineno, section, None, None])
                continue
            text = line.strip()
        elif not stripped[0].isspace():
            before, equals, after = stripped.partition("=")
            if not equals or ":" in before:
                before, colon, after = stripped.partition(":")
                if not colon:
                    raise ParseError(path, lineno, f"unexpected line: {stripped!r}")
            value = after.strip()
            records.append([lineno, section, before.strip(), _cut(value) if strip_inline_comments else value])
            continue
        else:
            text = _cut(line.strip()) if strip_inline_comments else line.strip()
        if not records or records[-1][2] is None:
            raise ParseError(path, lineno, "unexpected value continuation")
        records[-1][3] = f"{records[-1][3]}\\n{text}" if records[-1][3] else text
    sections, sources = {}, {}
    for lineno, section, name, value in records:
        if section is None:
            raise ParseError(path, lineno, "no section header defined")
        if name is None:
            if section in sections:
                raise ParseError(path, lineno, f"duplicate section {section!r}")
            sections[section] = {}
            sources[(section, None)] = lineno
        else:
            if name in sections[section]:
                raise ParseError(path, lineno, f"duplicate name {name!r}")
            sections[section][name] = value
            sources[(section, name)] = lineno
    return sections, sources
'''


def test_a_parser_written_from_the_two_pass_spec_passes_every_hidden_check():
    results = grade(answer(TWO_PASS), EXCEPTIONS)
    assert results["code"] and results["passed"] == results["total"] == len(CASES), [n for n, ok in results["results"].items() if not ok]


def test_the_two_pass_spec_replaces_only_the_specification_in_the_context(tmp_path):
    from scripts.live.iniparse_task import SPEC, SPEC_TWO_PASS, context
    for name in ("iniconfig/exceptions.py", "iniconfig/__init__.py"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text(f"# {name}\n")
    plain, two_pass = context(tmp_path), context(tmp_path, SPEC_TWO_PASS)
    assert SPEC.rstrip() in plain and SPEC_TWO_PASS.rstrip() in two_pass
    assert plain.split("### iniconfig/exceptions.py")[1] == two_pass.split("### iniconfig/exceptions.py")[1]
