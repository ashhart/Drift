"""Gate items keep the drop-in framing and draw a reproducible sample."""
import json
from scripts.live.gate_items import HEAD, items, tail


def _questions(path, name, kinds):
    questions = [{"id": f"{name}q{i}", "kind": kind, "question": f"Q{i}?", "answer": f"A{i}"} for i, kind in enumerate(kinds)]
    path.write_text(json.dumps({"context": f"### {name}.py\n```python\nx = 1\n```\n", "files": [f"{name}.py"], "questions": questions}))


def test_items_carry_the_framing_and_name_the_passage_by_file(tmp_path):
    _questions(tmp_path / "h1.json", "h1", ["raises", "signature", "default"])
    found = items([tmp_path / "h1.json"], {"raises", "default"})
    assert [i["id"] for i in found] == ["h1-h1q0", "h1-h1q2"]
    assert found[0]["head_text"] == HEAD and found[0]["tail_text"] == tail("Q0?")
    assert HEAD.endswith("Project files, from our shared memory:\n") and tail("Q0?").endswith("<think>\n\n</think>\n\n")
    assert found[0]["passage"].startswith("### h1.py")


def test_a_limit_keeps_a_seeded_sample_in_order(tmp_path):
    _questions(tmp_path / "h1.json", "h1", ["signature"] * 10)
    first, again = items([tmp_path / "h1.json"], {"signature"}, 4, seed=3), items([tmp_path / "h1.json"], {"signature"}, 4, seed=3)
    assert len(first) == 4 and first == again
    order = [int(i["id"].split("q")[-1]) for i in first]
    assert order == sorted(order)
