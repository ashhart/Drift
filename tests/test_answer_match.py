"""Answer keys match by kind: numbers by value, strings quoted, function names as whole identifiers, prose as words."""
from drift.eval.answer_match import matches, numbers

CONSTANT = "In packaging/_elffile.py, what value is assigned to EF_ARM_ABIMASK?"
DEFAULT = "In fsspec/parquet.py, what is the default value of the max_gap parameter of _get_parquet_byte_ranges?"
RAISES = 'Which function in drift/handoffd_client.py fails with the message "no result line for the pull"? Give the function name.'


def test_a_number_matches_by_value_in_any_spelling_but_not_inside_a_name_or_a_longer_number():
    assert matches("4278190080", CONSTANT, "The value assigned to `EF_ARM_ABIMASK` is `0xFF000000`.")
    assert matches("64000", DEFAULT, "It is **64,000**.") and matches("64000", DEFAULT, "max_gap=64_000")
    assert not matches("40", DEFAULT, "The default is **400** (or `400.0`).") and matches("40", DEFAULT, "It is 40.0.")
    assert not matches("5", CONSTANT, "EF_ARM_ABI_VER5 is set elsewhere")
    assert not matches("64000", DEFAULT, "It is **65536** (or `64_000` in the code representation).")
    assert not matches("64000", DEFAULT, "The `max_gap` parameter is **65536** (or `64_000`).\n```python\ndef f(max_gap=64_000):\n```")
    head_dim = "In drift/serving/foreign_positions.py, what is the default value of the head_dim parameter of __init__?"
    assert matches("256", head_dim, "```python\ndef __init__(self, layers, *, heads=2, head_dim=256):\n```\nIt is **256**.")
    assert matches("256", head_dim, "The parameter is `head_dim`:\n```python\ndef __init__(self, *, heads=2, head_dim=256):\n```")
    assert numbers("0x400, 1e3, 1,000,000 and VER5") == [1024, 1000.0, 1000000]


def test_a_string_value_must_be_quoted_so_an_echo_of_the_question_does_not_count():
    question = "In jinja2/idtracking.py, what value is assigned to VAR_LOAD_UNDEFINED?"
    assert not matches("undefined", question, "The value assigned to `VAR_LOAD_UNDEFINED` is `None`.")
    assert matches("undefined", question, 'VAR_LOAD_UNDEFINED = "undefined"') and matches("undefined", question, 'It is `"undefined"`.')
    assert matches("default", "In fsspec/generic.py, what is the default value of the default_method parameter of __init__?", "default")


def test_a_function_name_matches_as_a_whole_identifier_outside_the_echoed_message_and_path():
    assert matches("pull", RAISES, "The function is `pull`.")
    assert not matches("pull", RAISES, 'I cannot tell which function fails with "no result line for the pull".')
    assert not matches("_pages", 'Which function in drift/translate/dsv4_pages.py fails with the message "rows must cover whole pages"?', "It is `_load_pages`.")
    assert not matches("select", 'Which function in x.py fails with the message "no match"?', "It raises PylockSelectError.")
    assert matches("__init__", 'Which function in x.py fails with the message "no match"?', "It is `Package.__init__`.")


def test_other_keys_match_whole_words_with_a_plural_allowed_and_an_empty_key_never_matches():
    assert matches("dried apricot", "What is the special ingredient?", "a sack of dried apricots")
    assert not matches("red", "Which colour?", "Nothing was shared with me.") and matches("hare", "Which animal?", "A hare.")
    assert not matches("", "Write the module.", "anything at all")


def test_a_def_line_is_still_scored_by_parsing():
    key = "def pull(self, peer: str, offset: int=0):"
    assert matches(key, "Write the full def line of pull.", "```python\ndef pull(self, peer:str, offset:int = 0):\n```")
    assert not matches(key, "Write the full def line of pull.", "def pull(self, peer, offset=4096):")


def test_the_opening_of_a_def_line_used_as_a_training_key_is_matched_as_words():
    assert matches("def pull(", "Write out pull in the project files.", "def pull(self, peer):\n    return peer")
    assert not matches("def pull(", "Write out pull in the project files.", "def push(self):\n    pass")
