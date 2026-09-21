import json
import random

import pytest

from drift.eval.stream_repetition import RepetitionPolicy, StreamingRepetition


def test_repetition_detection_is_chunk_invariant_and_latched():
    text = "repeat these four words " * 100
    baseline = StreamingRepetition(RepetitionPolicy())
    assert baseline.feed(text, final=True)
    for seed in range(20):
        monitor = StreamingRepetition(RepetitionPolicy())
        rng = random.Random(seed)
        position = 0
        while position < len(text):
            stop = position + rng.randint(1, 37)
            monitor.feed(text[position:stop])
            position = stop
        monitor.feed("", final=True)
        assert monitor.report() == baseline.report()
        monitor.feed("different novel information")
        assert monitor.report() == baseline.report()


def test_unique_stream_state_is_bounded_and_diagnostics_have_no_text():
    monitor = StreamingRepetition(RepetitionPolicy(window_words=40))
    for i in range(5000):
        assert not monitor.feed(f"private-word-{i} ")
    assert not monitor.feed("", final=True)
    report = monitor.report()
    assert report["retained_words"] == 40 and report["pending_chars"] == 0
    assert report["words_seen"] == 5000 and report["rate"] == 0
    assert "private-word" not in json.dumps(report)
    with pytest.raises(ValueError, match="completion"):
        monitor.feed("late output")


def test_long_unbroken_output_aborts_without_unbounded_buffer():
    monitor = StreamingRepetition(RepetitionPolicy(max_word_chars=12))
    for _ in range(12):
        assert not monitor.feed("x")
    assert monitor.feed("x" * 10000)
    assert monitor.report()["reason"] == "word_size_limit"
    assert monitor.report()["pending_chars"] == 12


def test_final_word_and_stream_independence():
    policy = RepetitionPolicy(ngram=2, min_words=8, window_words=8, threshold=0.7)
    a, b = StreamingRepetition(policy), StreamingRepetition(policy)
    assert not a.feed("a b a b a b a b")
    assert a.feed("", final=True)
    assert not b.feed("one two three four five six seven eight", final=True)
    assert b.report()["rate"] == 0


@pytest.mark.parametrize("config", [{"threshold": float("nan")}, {"threshold": 0},
                                    {"min_words": 2}, {"window_words": 16}, {"max_word_chars": 0}])
def test_invalid_policy_is_rejected(config):
    with pytest.raises(ValueError):
        RepetitionPolicy(**config)
