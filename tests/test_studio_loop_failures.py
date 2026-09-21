"""Exercise the worker's real publication functions without loading model weights."""
import ast
import io
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest


SOURCE = Path(__file__).parents[1] / "scripts/live/studio_mcdma_loop.py"


def worker_function(name, namespace):
    tree = ast.parse(SOURCE.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


class BrokenPublisher:
    def deliver(self, *args):
        raise TimeoutError("synthetic transport timeout")

    def poll_applied(self, *args):
        raise ValueError("synthetic rank receipt mismatch")

    confirm_applied = poll_applied


def test_publication_failure_stops_the_worker_instead_of_silent_unlinked_decode():
    publisher = BrokenPublisher()
    namespace = dict(
        published_upto=0, sequence=0, glm_used=0, own_slots=[0], own_positions=[0],
        share=None, publisher=publisher, cache=None, rope=None, QL=(1,), GL=(3,),
        args=SimpleNamespace(prompt_only=False, no_reverse=False, prompt_copies=None,
                             copies=1, glm_reserve=64, session="synthetic", sync_confirm=False),
        time=time, np=np, io=io, unconfirmed=[],
        tap_slots=lambda *args: {1: (np.zeros((1, 1, 1)), np.zeros((1, 1, 1)))},
        rev=SimpleNamespace(read=lambda *args: {3: np.zeros((1, 512))}),
    )
    publish = worker_function("publish_own", namespace)
    with pytest.raises(RuntimeError, match="publication failed"):
        publish(0)
    assert namespace["publisher"] is publisher
    assert namespace["sequence"] == namespace["glm_used"] == 0


@pytest.mark.parametrize("block", [False, True])
def test_rank_receipt_failure_stops_instead_of_discarding_the_pending_publication(block):
    pending = [({"sequence": 0}, 1, {"sequence": 0})]
    namespace = dict(publisher=BrokenPublisher(), unconfirmed=pending)
    confirm = worker_function("confirm_pending", namespace)
    with pytest.raises(RuntimeError, match="confirmation failed"):
        confirm(block)
    assert len(pending) == 1
