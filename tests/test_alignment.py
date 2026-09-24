"""Pairing two tokenizations of one text by the characters their tokens end on."""
import numpy as np
from drift.translate.alignment import aligned_pairs


def test_tokens_ending_on_the_same_character_are_paired_in_order():
    source = np.array([[0, 3], [3, 5], [5, 9], [9, 10]])                    # "def" "_x" "(sel" "f"
    target = np.array([[0, 3], [3, 4], [4, 5], [5, 10]])                    # "def" "_" "x" "(self"
    s, t = aligned_pairs(source, target)
    assert s.tolist() == [0, 1, 3] and t.tolist() == [0, 2, 3]


def test_no_shared_end_gives_empty_pairs():
    s, t = aligned_pairs(np.array([[0, 2]]), np.array([[0, 3]]))
    assert s.size == 0 and t.size == 0
