"""Marker salience and addressing training (spec M3.2), training-only Q/A tasks.

The sender writes a question on a private branch with its learned marker; the receiver
attends to the mail block plus native context and answers. Objectives:

  salience:  -log(mean receiver attention mass on the mail block)   (mail gets noticed)
  behavior:  KL(teacher || student) in the receiver vocabulary       (mail gets used)

Teacher = receiver with the question as legitimate text. Student = receiver with the
mail entries. Gradients reach the marker, the sender's writer, the receiver's reader
and the gates; backbones stay frozen and are asserted. Attention mass is a diagnostic
term here, never evidence of incorporation (D9).
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import torch
from drift.adapters.base import ForeignEntries
from drift.core.attention import Gate
from drift.core.types import KV
from drift.mailbox.stream import MailWriter
from drift.train.fit import receiver_kl
from drift.translate.pool import Translator


@dataclass(frozen=True)
class MailExample:
    sender_parent: torch.Tensor       # sender's private context ids
    question_sender: torch.Tensor     # the question in the sender tokenizer (locally authorized)
    question_receiver: torch.Tensor   # the same question in the receiver tokenizer (teacher only)
    receiver_context: torch.Tensor    # receiver's own native context
    answer_prefix: torch.Tensor       # receiver ids scored by both teacher and student


@dataclass
class MailTrainLedger:
    steps: int = 0
    sender_mail_tokens: int = 0
    receiver_teacher_tokens: int = 0
    receiver_student_tokens: int = 0
    wall_seconds: float = 0.0
    losses: list[dict] = field(default_factory=list)


def train_marker(sender: Any, receiver: Any, mail: MailWriter, writer: Translator, reader: Translator,
                 gates: Mapping[int, Gate], examples: Sequence[MailExample], steps: int, learning_rate: float,
                 lambda_salience: float = 1.0, lambda_kl: float = 1.0, seed: int = 0) -> MailTrainLedger:
    if not examples or steps <= 0 or learning_rate <= 0:
        raise ValueError("examples, positive steps and learning rate required")
    params = [mail.marker] + list(writer.writers.parameters()) + list(reader.readers.parameters()) + [g.logit for g in gates.values()]
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    ledger, started = MailTrainLedger(), time.perf_counter()
    digests = (sender.frozen_digest(), receiver.frozen_digest())
    for _ in range(steps):
        ex = examples[int(torch.randint(len(examples), (1,), generator=generator))]
        with torch.no_grad():
            parent = sender.forward(ex.sender_parent).state
            native = receiver.forward(ex.receiver_context)
            teacher = receiver.forward(torch.cat((ex.receiver_context, ex.question_receiver, ex.answer_prefix)))
            ledger.receiver_teacher_tokens += int(ex.receiver_context.numel() + ex.question_receiver.numel() + ex.answer_prefix.numel())
        canonical = mail.write(sender, parent, ex.question_sender)      # gradient into the marker
        ledger.sender_mail_tokens += int(ex.question_sender.numel())
        entries = reader.read(writer.write(canonical))
        foreign = ForeignEntries(torch.arange(next(iter(canonical.values())).tokens if isinstance(next(iter(canonical.values())), KV)
                                              else next(iter(canonical.values())).shape[0]), entries)
        student = receiver.forward(ex.answer_prefix, native.state, foreign=foreign, gates=gates)
        ledger.receiver_student_tokens += int(ex.answer_prefix.numel())
        mass = torch.stack([m.mean() for m in student.foreign_mass.values()]).mean()
        salience = -torch.log(mass.clamp_min(1e-9))
        n = ex.answer_prefix.numel()
        kl = receiver_kl(student.output, teacher.output[-n:])
        loss = lambda_salience * salience + lambda_kl * kl
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite mailbox training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        ledger.steps += 1
        ledger.losses.append({"loss": float(loss), "salience": float(salience), "kl": float(kl), "mass": float(mass)})
    ledger.wall_seconds = time.perf_counter() - started
    if (sender.frozen_digest(), receiver.frozen_digest()) != digests:
        raise RuntimeError("a backbone changed during mailbox training")
    return ledger
