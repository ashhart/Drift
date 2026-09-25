"""Def-line scoring: parsed comparison, callable against exact, and answers wrapped in prose or code blocks."""
import ast
from drift.eval.signatures import def_line, score

EXPECTED = "def pull(self, peer: str, remote_path: str, offset: int=0, unlink: bool=True) -> Pull:"


def test_the_def_line_drops_decorators_and_body():
    node = ast.parse("@staticmethod\ndef view(peer: str, offset: int, length: int) -> memoryview:\n    return None\n").body[0]
    assert def_line(node) == "def view(peer: str, offset: int, length: int) -> memoryview:"


def test_an_answer_in_a_code_block_with_other_spacing_is_exact():
    answer = "Here it is:\n```python\ndef pull(self, peer:str, remote_path:str, offset:int = 0, unlink:bool = True)->Pull:\n    ...\n```"
    assert score(answer, EXPECTED) == {"found": True, "callable": True, "exact": True}


def test_missing_annotations_are_callable_but_not_exact():
    assert score("def pull(self, peer, remote_path, offset=0, unlink=True):", EXPECTED) == {"found": True, "callable": True, "exact": False}


def test_a_wrong_default_or_order_is_neither():
    assert score("def pull(self, peer: str, remote_path: str, offset: int=4096, unlink: bool=True) -> Pull:", EXPECTED)["callable"] is False
    assert score("def pull(self, remote_path: str, peer: str, offset: int=0, unlink: bool=True) -> Pull:", EXPECTED)["callable"] is False


def test_a_def_split_over_lines_is_joined_and_no_def_is_not_found():
    answer = "def pull(\n    self,\n    peer: str,\n    remote_path: str,\n    offset: int = 0,\n    unlink: bool = True,\n) -> Pull:"
    assert score(answer, EXPECTED)["exact"] is True
    assert score("It takes a peer and a path.", EXPECTED) == {"found": False, "callable": False, "exact": False}


def test_a_def_header_longer_than_eight_lines_is_found():
    header = "def select(\n    self,\n    *,\n" + "".join(f"    p{i}: int = {i},\n" for i in range(12)) + ") -> None:\n    pass\n"
    expected = "def select(self, *, " + ", ".join(f"p{i}: int={i}" for i in range(12)) + ") -> None:"
    assert score(f"```python\n{header}```", expected)["callable"]
