"""Window questions find whole functions line by line in text cut mid-file, and ask only what the window settles."""
import pytest
from drift.eval.signatures import score
from drift.eval.window_questions import functions, questions

WINDOW = '''        return self.value          # the tail of a method cut at the window start
### pkg/pages.py
```python
def pages(rows: int, per_page: int = 64) -> int:
    if rows % per_page:
        raise ValueError("rows must cover whole pages")
    total = rows // per_page
    return total


class Reader:
    def read(self, blob,
             name, *, rank=0):
        data = blob[name]
        if rank < 0:
            raise ValueError(f"rank {rank} is negative")
        return data


def cut(stream, limit):
    for line in stream:
        yield line[:limit]
'''


def test_functions_parse_multiline_headers_and_mark_bodies_cut_by_the_window():
    found = {f["name"]: f for f in functions(WINDOW)}
    assert set(found) == {"pages", "read", "cut"}
    assert found["pages"]["path"] == "pkg/pages.py" and found["pages"]["complete"]
    assert score(found["pages"]["def_line"], "def pages(rows: int, per_page: int = 64) -> int:")["exact"]
    assert score(found["read"]["def_line"], "def read(self, blob, name, *, rank=0):")["exact"] and found["read"]["complete"]
    assert not found["cut"]["complete"]                                   # its body runs to the end of the window


def test_kinds_ask_only_what_the_window_settles():
    qs = questions(WINDOW)
    kinds = {(q["kind"], q["answer"]) for q in qs}
    assert ("signature", "def pages(") in kinds and ("signature", "def read(") in kinds and ("signature", "def cut(") in kinds
    assert ("recall", "def pages(") in kinds and ("recall", "def read(") in kinds and ("recall", "def cut(") not in kinds
    assert ("raises", "pages") in kinds and not any(k == "raises" and a == "read" for k, a in kinds)   # f-string messages are skipped
    recall = next(q for q in qs if q["kind"] == "recall" and q["answer"] == "def pages(")
    assert recall["question"].startswith("Write out pages in pkg/pages.py exactly as the file has it")
    follow = [q for q in qs if q["kind"] == "continue"]
    assert follow and all(len(q["answer"]) >= 20 for q in follow)
    assert any(q["answer"] == "total = rows // per_page" and q["question"].endswith('raise ValueError("rows must cover whole pages")') for q in follow)


def test_names_defined_twice_are_skipped_and_unknown_kinds_refused():
    twice = WINDOW + "\n\ndef pages(a, b):\n    return a\n    b\n    a\n    b\n\nX = 1\n"
    assert not [q for q in questions(twice, ("signature", "recall")) if "pages" in q["answer"]]
    with pytest.raises(ValueError, match="unknown kinds"):
        questions(WINDOW, ("summary",))


def test_a_block_asks_for_the_eight_lines_after_a_distinctive_line_and_stays_inside_the_window():
    lines = [f"value_{i} = compute_something_long({i}, factor={i * 3})" for i in range(30)]
    window = "\n".join(lines) + "\n"
    blocks = [q for q in questions(window, ("block",))]
    assert blocks and all(q["question"].startswith("In the project files, quote the 8 lines") for q in blocks)
    first = next(q for q in blocks if q["question"].endswith(lines[0]))
    assert first["answer"] == lines[1]
    assert any(q["question"].endswith(lines[21]) for q in blocks)       # lines 22 to 29 all lie in the window
    assert not any(q["question"].endswith(lines[22]) for q in blocks)   # only seven lines follow line 22


def test_a_table_asks_what_a_row_says_under_another_column():
    window = ("Intro line.\n| Field | Rule | Example |\n|-------|------|---------|\n| sku | 1-64 uppercase letters, digits, underscores or hyphens | BOLT-M8 |\n"
              "| event_id | letters, digits, dots, underscores or hyphens | public-1 |\n| sku | duplicate key row | X |\nAfter.\n")
    qs = questions(window, ("table",))
    assert {q["answer"] for q in qs} == {"letters, digits, dots, underscores or hyphens", "public-1"}   # the sku rows repeat their key
    q = next(q for q in qs if q["answer"] == "public-1")
    assert q["question"] == ("In the project files, a table with the columns Field, Rule, Example has a row for event_id. "
                             "What does that row say under Example? Quote it exactly.")
