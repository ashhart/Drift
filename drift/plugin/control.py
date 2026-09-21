"""Typed reference control schema, NOT a claim about OMP's actual plugin API."""
from dataclasses import dataclass
from enum import IntEnum
from uuid import UUID


class Op(IntEnum):
    START = 1
    TICK = 2
    PAUSE = 3
    ABORT = 4
    COMPLETE = 5


@dataclass(frozen=True)
class ControlEvent:
    run: UUID
    op: Op
    epoch: int

    @classmethod
    def parse(cls, event: dict) -> "ControlEvent":
        if set(event) != {"run", "op", "epoch"}:
            raise ValueError("control plane rejects free text, hints, token IDs and extra fields")
        if type(event["epoch"]) is not int or event["epoch"] < 0:
            raise ValueError("invalid epoch")
        if type(event["op"]) is not int:
            raise ValueError("op must be a fixed numeric enum")
        return cls(UUID(event["run"]), Op(event["op"]), event["epoch"])
