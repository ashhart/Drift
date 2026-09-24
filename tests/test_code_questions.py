"""Code questions come from the source itself; recall is opt-in and asks for whole functions."""
import pytest
from drift.eval.code_questions import ALL_KINDS, context_text, questions, split_context

SOURCE = '''LIMIT = 4096


def pages(rows, per_page=64):
    if rows % per_page:
        raise ValueError("rows must cover whole pages")
    return rows // per_page


async def fetch(url, retries=5, *, timeout=30.0):
    for attempt in range(retries):
        pass
    return url


def short(a, b):
    return a
'''


def test_context_round_trips_and_malformed_context_is_refused():
    files = [("pkg/a.py", SOURCE), ("pkg/b.py", "X = 1\n")]
    context = context_text(files)
    assert context.startswith("### pkg/a.py\n```python\nLIMIT = 4096") and split_context(context) == [("pkg/a.py", SOURCE.rstrip()), ("pkg/b.py", "X = 1")]
    with pytest.raises(ValueError, match="path headers"):
        split_context(context + "trailing prose")


def test_default_kinds_leave_recall_out_and_recall_asks_for_whole_functions():
    files = [("pkg/a.py", SOURCE)]
    plain = questions(files)
    assert {q["kind"] for q in plain} == {"constant", "default", "raises", "signature"}
    assert any(q["kind"] == "raises" and q["answer"] == "pages" for q in plain)
    recall = [q for q in questions(files, ALL_KINDS) if q["kind"] == "recall"]
    assert [q["answer"] for q in recall] == ["def pages(", "async def fetch("]         # short() has too few lines
    assert recall[0]["question"].startswith("Write out pages from pkg/a.py exactly as the file has it")
    only = questions(files, ("recall",))
    assert only == recall


def test_names_defined_twice_are_not_asked_and_unknown_kinds_are_refused():
    twice = [("a.py", SOURCE), ("b.py", SOURCE)]
    assert not [q for q in questions(twice, ALL_KINDS) if q["kind"] in ("recall", "signature")]
    with pytest.raises(ValueError, match="unknown kinds"):
        questions(twice, ("recall", "summary"))
