"""The prefill-cost scorer must refuse uninstrumented rows and report an honest verdict."""
import json

import pytest

from scripts.score_prefill_cost import load, score


def rows(memory, text, passages=8):
    return [{"passage": f"p{index % passages}", "kind": "code",
             "first_token_s": {"drift_v4": memory, "text_in_prompt": text},
             "transport": {"translate_both_s": 0.05, "rdma_loop_s": 0.018}} for index in range(passages * 2)]


def test_a_clear_saving_is_supported():
    report = score(rows(0.04, 0.5))
    assert report["text_over_memory_ratio"] == 12.5
    assert report["P1_receiver_prefill_is_cheaper"] and report["P2_saving_is_material"]
    assert report["verdict"] == "SUPPORTED"


def test_a_real_but_small_saving_is_only_partly():
    report = score(rows(0.10, 0.15))
    assert report["P1_receiver_prefill_is_cheaper"] and not report["P2_saving_is_material"]
    assert report["verdict"] == "PARTLY"


def test_text_being_faster_is_reported_not_hidden():
    report = score(rows(0.5, 0.04))
    assert report["questions_where_memory_was_faster"] == 0
    assert report["verdict"] == "NOT SUPPORTED"


def test_uninstrumented_rows_are_refused(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"passage": "p0", "transport": {}}) + "\n")
    with pytest.raises(SystemExit):
        load(path)
