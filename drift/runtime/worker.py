"""Stateful local worker: one member of a Drift/Hive session (spec M2.1, D15).

The worker owns its adapter, private native state, translator, incoming pool bank,
publication cursor and counters. `step` pins the incoming view for the whole step,
runs one adapter forward with the pinned entries as foreign memory, retains the new
private state, and returns a Publication of only its new canonical rows in pool
format. Local tokens never leave the worker; only pool rows do.

Native source position, publication sequence and coupling epoch are distinct
counters (a prefill consumes many positions in one epoch).
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Any, Mapping
from uuid import UUID
import torch
from drift.adapters.base import ForeignEntries, StepOutput
from drift.core.attention import Gate
from drift.core.pool import PoolBank
from drift.mailbox.stream import MailWriter, mail_publication, merge_entries
from drift.translate.pool import Translator
from drift.transport.wire2 import Publication


@dataclass
class WorkerCounters:
    epoch: int = -1
    sequence: int = 0
    source_position: int = 0
    local_tokens: int = 0
    steps: int = 0
    mail_sequence: int = 0
    mail_position: int = 0
    mail_tokens: int = 0


@dataclass
class StepReport:
    epoch: int
    output: torch.Tensor
    publication: Publication
    foreign_tokens: int
    foreign_mass: Mapping[int, torch.Tensor]
    canonical_norms: dict[int, float] = field(default_factory=dict)
    entry_mass: Mapping[int, torch.Tensor] = field(default_factory=dict)   # over native-then-mail entries
    native_foreign_tokens: int = 0
    mail_foreign_tokens: int = 0


class Worker:
    def __init__(self, member: str, writer: int, session: UUID, adapter: Any, translator: Translator,
                 bank: PoolBank, gates: Mapping[int, Gate] | None = None, override: float | None = None,
                 seed: int = 0, mail_writer: int | None = None, mail_bank: PoolBank | None = None,
                 mail: MailWriter | None = None):
        if bank.self_writer != writer:
            raise ValueError("the worker's bank must exclude the worker's own writer id")
        if translator.member != member:
            raise ValueError("translator belongs to another member")
        if (mail_writer is None) != (mail_bank is None) or (mail_writer is None) != (mail is None):
            raise ValueError("mail needs a mail writer id, a mail bank and a MailWriter together")
        if mail_bank is not None and (mail_bank.self_writer != mail_writer or mail_writer == writer):
            raise ValueError("the mail stream needs its own writer id, excluded from its own mail bank")
        if mail_bank is not None and not mail_bank.multiple_per_epoch:
            raise ValueError("a mail bank must allow several publications per epoch (multiple_per_epoch=True)")
        self.member, self.writer, self.session = member, writer, session
        self.adapter, self.translator, self.bank = adapter, translator, bank
        self.mail_writer, self.mail_bank, self.mail = mail_writer, mail_bank, mail
        self.gates, self.override = gates, override
        self.state: Any = None
        self.counters = WorkerCounters()
        self.rng = random.Random(seed)
        self.poisoned = False

    def step(self, local_ids: torch.Tensor, epoch: int) -> StepReport:
        if self.poisoned:
            raise RuntimeError("worker is poisoned; restore a checkpoint or start a new session")
        if epoch != self.counters.epoch + 1:
            raise ValueError("epochs must advance by exactly one")
        try:
            view = self.bank.pin()
            mail_view = None if self.mail_bank is None else self.mail_bank.pin()
            for v in (view, mail_view):
                if v is not None and v.epoch >= epoch:
                    raise ValueError("incoming view is not from a prior epoch; lockstep violated")
            foreign = merge_entries(view, mail_view)
            out: StepOutput = self.adapter.forward(local_ids, self.state, foreign, self.gates, self.override)
            rows = self.translator.write(out.canonical)
            rows = {level: r.detach() for level, r in rows.items()}
            pub = Publication(self.session, self.writer, self.translator.pool.fingerprint, epoch,
                              self.counters.sequence, self.counters.source_position, rows)
            pub.check()
        except Exception:
            self.poisoned = True
            raise
        # State advances only after everything above succeeded.
        self.state = out.state
        self.counters.epoch = epoch
        self.counters.sequence += 1
        self.counters.source_position += int(local_ids.numel())
        self.counters.local_tokens += int(local_ids.numel())
        self.counters.steps += 1
        norms = {}
        for layer, entry in out.canonical.items():
            tensor = entry.k if hasattr(entry, "k") else entry
            norms[layer] = float(tensor.detach().float().norm(dim=-1).mean())
        native_n = 0 if view is None else int(view.positions.numel())
        mail_n = 0 if mail_view is None else int(mail_view.positions.numel())
        return StepReport(epoch, out.output, pub, native_n + mail_n,
                          {k: v.detach() for k, v in out.foreign_mass.items()}, norms,
                          {k: v.detach() for k, v in out.entry_mass.items()}, native_n, mail_n)

    def send_mail(self, local_ids: torch.Tensor, epoch: int) -> Publication:
        """Write locally authorized tokens plus the marker on a private branch; publish
        only the branch's canonical rows on the mail stream. The native state is untouched."""
        if self.mail is None:
            raise ValueError("this worker has no mail stream")
        if self.poisoned:
            raise RuntimeError("worker is poisoned")
        if epoch != self.counters.epoch:
            raise ValueError("mail is sent in the epoch of the last completed step")
        canonical = self.mail.write(self.adapter, self.state, local_ids)
        pub = mail_publication(self.session, self.mail_writer, self.translator, epoch,
                               self.counters.mail_sequence, self.counters.mail_position, canonical)
        pub.check()
        self.counters.mail_sequence += 1
        self.counters.mail_position += pub.tokens
        self.counters.mail_tokens += int(local_ids.numel())
        return pub

    def checkpoint(self) -> dict:
        return {
            "member": self.member, "writer": self.writer, "session": self.session,
            "counters": WorkerCounters(**vars(self.counters)),
            "state": None if self.state is None else self.adapter.snapshot_state(self.state),
            "bank": self.bank.snapshot(), "rng": self.rng.getstate(), "poisoned": self.poisoned,
            "mail_bank": None if self.mail_bank is None else self.mail_bank.snapshot(),
        }

    def restore(self, checkpoint: dict) -> None:
        if (checkpoint["member"], checkpoint["writer"], checkpoint["session"]) != (self.member, self.writer, self.session):
            raise ValueError("checkpoint belongs to another worker or session")
        if checkpoint["poisoned"]:
            raise ValueError("refusing to restore a poisoned checkpoint")
        if checkpoint["state"] is None:
            self.state = None
        else:
            if self.state is None:
                raise ValueError("restoring native state needs a live template state of the same shape")
            self.state = self.adapter.restore_state(self.state, checkpoint["state"])
        self.bank.restore(checkpoint["bank"])
        if self.mail_bank is not None:
            self.mail_bank.restore(checkpoint["mail_bank"])
        self.counters = WorkerCounters(**vars(checkpoint["counters"]))
        self.rng.setstate(checkpoint["rng"])
        self.poisoned = False
