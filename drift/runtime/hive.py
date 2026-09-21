"""N-member one-epoch-lag scheduler over pool banks (spec M2.2 generalized; H8).

At epoch k every member pins its incoming view (committed at k-1), every member
computes, and only then are the epoch-k publications delivered to every other
member through the transport codec. No member ever sees a same-epoch publication.
Any failure poisons the controller; resume from worker checkpoints, never in place.
The reference computes members serially; a qualified implementation may run them
concurrently because none consumes same-epoch output.
"""
from __future__ import annotations
from typing import Callable, Mapping
import torch
from drift.runtime.worker import StepReport, Worker
from drift.transport.wire2 import FrameCodec2, Publication


class HiveController:
    def __init__(self, workers: Mapping[str, Worker], codec: FrameCodec2,
                 deliver: Callable[[bytes], bytes] | None = None):
        if len(workers) < 2:
            raise ValueError("a hive needs at least two members")
        ids = [w.writer for w in workers.values()]
        if len(set(ids)) != len(ids):
            raise ValueError("writer ids must be unique")
        for name, worker in workers.items():
            if worker.member != name:
                raise ValueError("worker name mismatch")
            if worker.bank.writers != frozenset(ids) - {worker.writer}:
                raise ValueError(f"{name}'s bank must enroll exactly the other members")
        self.workers, self.codec = dict(workers), codec
        # The transport hop: bytes in, bytes out. Identity models inproc; tests inject faults.
        self.deliver = deliver or (lambda frame: frame)
        self.epoch, self.failed = 0, False

    def tick(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, StepReport]:
        if self.failed:
            raise RuntimeError("controller is poisoned; restore checkpoints")
        if set(inputs) != set(self.workers):
            raise ValueError("every member consumes local input each epoch")
        try:
            for worker in self.workers.values():
                view = worker.bank.pin()
                if view is not None and view.epoch != self.epoch - 1:
                    raise ValueError("lockstep requires precisely the prior epoch")
                mail_view = None if worker.mail_bank is None else worker.mail_bank.pin()
                if mail_view is not None and mail_view.epoch >= self.epoch:
                    raise ValueError("mail from the current epoch cannot be read yet")
            reports = {name: worker.step(inputs[name], self.epoch) for name, worker in self.workers.items()}
            frames = {name: self.codec.encode(report.publication) for name, report in reports.items()}
            for source, frame in frames.items():
                for target, worker in self.workers.items():
                    if target == source:
                        continue
                    received: Publication = self.codec.decode(self.deliver(frame))
                    worker.bank.commit(received)
            self.epoch += 1
            return reports
        except Exception:
            self.failed = True
            raise

    def post_mail(self, sender: str, local_ids: torch.Tensor, mailbox=None, message_id: int | None = None) -> Publication:
        """After a tick: the sender writes mail for the just-completed epoch; every other
        member's mail bank receives it and will pin it from the next epoch on. With a
        MailboxController, the pool positions are registered per receiver before commit."""
        if self.failed:
            raise RuntimeError("controller is poisoned; restore checkpoints")
        if (mailbox is None) != (message_id is None):
            raise ValueError("register mail with both a controller and a message id")
        try:
            worker = self.workers[sender]
            pub = worker.send_mail(local_ids, self.epoch - 1)
            frame = self.codec.encode(pub)
            for name, other in self.workers.items():
                if name == sender:
                    continue
                if other.mail_bank is None:
                    raise ValueError(f"{name} has no mail bank")
                received = self.codec.decode(self.deliver(frame))
                if mailbox is not None:
                    mailbox.register_publication(name, received.writer, message_id, received, other.mail_bank)
                other.mail_bank.commit(received)
            return pub
        except Exception:
            self.failed = True
            raise

    def checkpoint(self) -> dict:
        return {"epoch": self.epoch, "workers": {name: w.checkpoint() for name, w in self.workers.items()}}

    def restore(self, checkpoint: dict) -> None:
        if set(checkpoint["workers"]) != set(self.workers):
            raise ValueError("checkpoint member set differs")
        for name, worker in self.workers.items():
            worker.restore(checkpoint["workers"][name])
        self.epoch, self.failed = checkpoint["epoch"], False
