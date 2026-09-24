"""The own-cache gate scorer counts each arm over the items that ran it."""
from scripts.score_state_gate import score


def item(answer, **arms):
    return {"answer": answer, **{arm: {"text": text} for arm, text in arms.items()}}


def test_an_arm_some_items_skipped_is_scored_over_the_rest():
    results = [item("Ada", text="Ada", rows="Ada", d_state="Ada Lovelace"), item("Bo", text="Bo", rows="no"),
               item("Cy", text="Cy.", rows="Cy.", d_state="unknown")]
    report = score(results)
    assert report["items"] == 3 and report["text_hits"] == 3 and report["rows_hits"] == 2
    assert report["d_state_items"] == 2 and report["d_state_hits"] == 1 and "rows_items" not in report
    assert report["rows_identical_to_text"] == 2 and report["d_state_identical_to_text"] == 0


def test_a_def_line_answer_is_scored_by_parsing():
    results = [item("def pull(self, peer: str, offset: int=0):", text="```python\ndef pull(self, peer:str, offset:int = 0):\n```", rows="def pull(self, peer, offset=4096):")]
    report = score(results)
    assert report["text_hits"] == 1 and report["rows_hits"] == 0


def test_a_run_without_the_text_arm_is_scored_without_comparing_to_it():
    report = score([item("Ada", t_rows="Ada", t_rows_state="no"), item("Bo", t_rows="Bo", t_rows_state="Bo")])
    assert report["t_rows_hits"] == 2 and report["t_rows_state_hits"] == 1
    assert "t_rows_identical_to_text" not in report and "text_hits" not in report


def test_an_arm_a_gate_skipped_is_left_out_of_its_count():
    results = [{"answer": "x", "text": {"text": "x"}, "t_rows_state": {"skipped": "too large"}},
               {"answer": "x", "text": {"text": "x"}, "t_rows_state": {"text": "x"}}]
    report = score(results)
    assert report["t_rows_state_hits"] == 1 and report["t_rows_state_items"] == 1

