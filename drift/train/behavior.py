"""Receiver-behavior training (spec M1.2) around adapters and translators.

  Donor:   frozen, source-private training context -> detached canonical KV.
  Map:     trainable writer (donor) and reader (receiver) sidecars -> receiver-space entries.
  Teacher: frozen RECEIVER with its legitimate context + continuation -> logits.
  Student: frozen RECEIVER with foreign entries + the same continuation -> logits.
  Loss:    lambda_reg * (regression of reader(writer(x_donor)) vs receiver canonical at aligned rows)
         + lambda_kl  * KL(teacher || student) on the receiver vocabulary at the same positions.

Only writer/reader parameters and gates receive gradients; backbones are frozen and
asserted so. Budgets are ledgered separately from regression-only fitting. Loss
weights, steps and selection rules are preregistered before any held-out evaluation.
"""
from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import torch
from drift.adapters.base import ForeignEntries
from drift.core.attention import Gate
from drift.core.types import KV
from drift.train.fit import receiver_kl
from drift.translate.pool import Translator


@dataclass(frozen=True)
class BehaviorExample:
    donor_context: torch.Tensor          # ids in the donor tokenizer
    receiver_context: torch.Tensor       # legitimate receiver ids for the teacher
    continuation: torch.Tensor           # receiver ids scored by both teacher and student
    aligned_rows: Sequence[tuple[int, int]] = ()   # (donor row, receiver row) causal endpoints for regression


@dataclass
class Preregistration:
    lambda_reg: float
    lambda_kl: float
    steps: int
    learning_rate: float
    seed: int
    stop_rule: str = "fixed_steps"
    notes: str = ""

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(vars(self), sort_keys=True).encode()).hexdigest()


@dataclass
class BehaviorLedger:
    donor_prefill_tokens: int = 0
    receiver_teacher_tokens: int = 0
    receiver_student_tokens: int = 0
    optimizer_steps: int = 0
    wall_seconds: float = 0.0
    losses: list[dict] = field(default_factory=list)


def _receiver_entries(writer: Translator, reader: Translator, donor_canonical: Mapping[int, KV | torch.Tensor]) -> Mapping[int, KV | torch.Tensor]:
    return reader.read(writer.write(donor_canonical))


def train_behavior(donor: Any, receiver: Any, writer: Translator, reader: Translator,
                   gates: Mapping[int, Gate], examples: Sequence[BehaviorExample],
                   prereg: Preregistration) -> BehaviorLedger:
    if not examples or prereg.steps <= 0 or prereg.learning_rate <= 0:
        raise ValueError("examples, positive steps and learning rate required")
    if writer.member == reader.member:
        raise ValueError("behavior training maps a donor member into a different receiver member")
    params = list(writer.writers.parameters()) + list(reader.readers.parameters()) + [g.logit for g in gates.values()]
    optimizer = torch.optim.AdamW(params, lr=prereg.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(prereg.seed)
    ledger, started = BehaviorLedger(), time.perf_counter()
    receiver_digest = receiver.frozen_digest()
    for _ in range(prereg.steps):
        example = examples[int(torch.randint(len(examples), (1,), generator=generator))]
        with torch.no_grad():
            donor_out = donor.forward(example.donor_context)
            ledger.donor_prefill_tokens += int(example.donor_context.numel())
            teacher = receiver.forward(torch.cat((example.receiver_context, example.continuation)))
            ledger.receiver_teacher_tokens += int(example.receiver_context.numel() + example.continuation.numel())
            receiver_canonical = receiver.forward(example.receiver_context).canonical
        donor_canonical = {k: (v.clone(detach=True) if isinstance(v, KV) else v.detach()) for k, v in donor_out.canonical.items()}
        entries = _receiver_entries(writer, reader, donor_canonical)
        foreign = ForeignEntries(torch.arange(example.donor_context.numel()), entries)
        student = receiver.forward(example.continuation, foreign=foreign, gates=gates)
        ledger.receiver_student_tokens += int(example.continuation.numel())
        n = example.continuation.numel()
        kl = receiver_kl(student.output, teacher.output[-n:])
        reg = torch.zeros((), dtype=torch.float32)
        if example.aligned_rows:
            d_rows = torch.tensor([d for d, _ in example.aligned_rows])
            r_rows = torch.tensor([r for _, r in example.aligned_rows])
            for layer, entry in entries.items():
                target = receiver_canonical[layer]
                if isinstance(entry, KV):
                    reg = reg + (entry.k[d_rows] - target.k[r_rows]).pow(2).mean() + (entry.v[d_rows] - target.v[r_rows]).pow(2).mean()
                else:
                    reg = reg + (entry[d_rows] - target[r_rows]).pow(2).mean()
        loss = prereg.lambda_reg * reg + prereg.lambda_kl * kl
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        ledger.optimizer_steps += 1
        ledger.losses.append({"loss": float(loss), "kl": float(kl), "reg": float(reg)})
    ledger.wall_seconds = time.perf_counter() - started
    if receiver.frozen_digest() != receiver_digest:
        raise RuntimeError("receiver backbone changed during training")
    return ledger
