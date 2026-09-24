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

