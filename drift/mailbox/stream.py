"""Mailbox streams over the pool (spec M3.1–M3.4, D21).

`MailWriter` is the token-conditioned v0 writer for any adapter: locally authorized
token IDs plus a learned marker go through a PRIVATE branch of the sender (its
private state is deep-copied and never retained), and only the branch's canonical
entries come out. It counts the local tokens it consumed.

`MailboxController` owns addressing. It is the controller, not the models: it
assigns message ids, allocates bounded slots, enforces TTL and a per-epoch creation
cap, tracks fixed statuses, and reconciles attention observations. Mail travels on
its own writer ids (one mail stream per sender) into a separate mail bank on each
receiver, so native cursors are never reused for private branches.

Everything here is mechanics and diagnostics. Attention on a message is not
delivery; incorporation is established only by counterfactual replay (eval/replay).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping
import torch
from torch import nn
from drift.adapters.base import ForeignEntries
from drift.core.pool import PoolBank, PoolView
from drift.core.types import KV
from drift.translate.pool import Translator
from drift.transport.wire2 import Publication


class MailWriter(nn.Module):
    def __init__(self, hidden_size: int, max_tokens: int = 32, marker_ple_id: int | None = None):
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(1, hidden_size))
        self.max_tokens, self.marker_ple_id = max_tokens, marker_ple_id
        self.local_tokens_consumed = 0
        self.forward_passes = 0

    def write(self, adapter: Any, private_parent: Any, local_ids: torch.Tensor) -> Mapping[int, KV | torch.Tensor]:
        if local_ids.ndim != 1 or not 1 <= local_ids.numel() <= self.max_tokens:
            raise ValueError("mail token budget exceeded")
        embeddings = adapter.embed(local_ids)
        embeddings = torch.cat((self.marker.to(embeddings), embeddings))
        ple = None
        if self.marker_ple_id is not None:
            ple = torch.cat((torch.tensor([self.marker_ple_id], dtype=torch.long), local_ids.to(torch.long)))
        out = adapter.forward_embeddings(embeddings, private_parent, ple_ids=ple)
        self.local_tokens_consumed += int(local_ids.numel())
        self.forward_passes += 1
        return dict(out.canonical)     # marker row first; the parent state is untouched


class MailStatus(str, Enum):
    PENDING = "pending"                    # created, not yet in any receiver view
    VISIBLE = "visible"                    # pinned by a receiver
    ATTENDED = "attended_not_proven"
    INCORPORATED = "causally_incorporated"
    EXPIRED = "expired_without_incorporation"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class MailRecord:
    id: int
    sender: str
    slot: int
    created: int
    expires: int
    tokens: int
    status: MailStatus = MailStatus.PENDING
    visible: int | None = None
    first_attention: int | None = None
    incorporation: int | None = None
    max_mass: float = 0.0


@dataclass
class MailboxController:
    """Controller-owned addressing. IDs, slots, TTL and caps are policy, never model text."""
    ttl: int
    slots: int
    per_epoch_cap: int
    attention_threshold: float = 0.01
    records: dict[int, MailRecord] = field(default_factory=dict)
    slot_owner: dict[int, int] = field(default_factory=dict)          # slot -> message id
    created_in_epoch: dict[int, int] = field(default_factory=dict)
    next_id: int = 0
    # pool-slot bookkeeping per receiver: (receiver, mail writer) -> list of (pool position, message id)
    positions: dict[tuple[str, int], list[tuple[int, int]]] = field(default_factory=dict)

    def create(self, sender: str, epoch: int, tokens: int) -> MailRecord:
        if self.ttl <= 0 or self.slots <= 0 or self.per_epoch_cap <= 0 or tokens <= 0:
            raise ValueError("invalid mailbox policy")
        if self.created_in_epoch.get(epoch, 0) >= self.per_epoch_cap:
            raise OverflowError("per-epoch mail creation cap reached")
        free = next((s for s in range(self.slots) if s not in self.slot_owner), None)
        if free is None:
            raise OverflowError("mailbox storm/backpressure: no free slot")
        record = MailRecord(self.next_id, sender, free, epoch, epoch + self.ttl, tokens)
        self.records[record.id] = record
        self.slot_owner[free] = record.id
        self.created_in_epoch[epoch] = self.created_in_epoch.get(epoch, 0) + 1
        self.next_id += 1
        return record

    def register_publication(self, receiver: str, writer: int, message_id: int, pub: Publication, bank: PoolBank) -> None:
        """Map the pool positions a mail publication will occupy in `receiver`'s mail bank."""
        n = pub.tokens
        start = bank.next_slot
        self.positions.setdefault((receiver, writer), []).extend((start + j, message_id) for j in range(n))

    def observe(self, receiver: str, epoch: int, view: PoolView | None, entry_mass: Mapping[int, torch.Tensor]) -> None:
        """Advance VISIBLE/ATTENDED from a receiver's pinned mail view and per-entry mass."""
        if view is None:
            return
        by_position: dict[int, int] = {}
        for (rcv, _), pairs in self.positions.items():
            if rcv == receiver:
                by_position.update(dict(pairs))
        mass = None
        if entry_mass:
            stacked = torch.stack([m.detach().float() for m in entry_mass.values()])
            mass = stacked.mean(0)
        for i, position in enumerate(view.positions.tolist()):
            message_id = by_position.get(position)
            if message_id is None:
                continue
            record = self.records[message_id]
            if record.status in {MailStatus.INCORPORATED, MailStatus.EXPIRED, MailStatus.REJECTED, MailStatus.CANCELLED}:
                continue
            if record.status == MailStatus.PENDING:
                record.status, record.visible = MailStatus.VISIBLE, epoch
            if mass is not None:
                value = float(mass[i])
                record.max_mass = max(record.max_mass, value)
                if value >= self.attention_threshold and record.status == MailStatus.VISIBLE:
                    record.status, record.first_attention = MailStatus.ATTENDED, epoch

    def _release(self, record: MailRecord) -> None:
        if self.slot_owner.get(record.slot) == record.id:
            del self.slot_owner[record.slot]

    def record_causal_use(self, message_id: int, epoch: int, *, active_correct: bool,
                          ablated_correct: bool, replay_started_before_exposure: bool) -> None:
        record = self.records[message_id]
        if not record.created <= epoch < record.expires:
            raise ValueError("outside delivery window")
        if not replay_started_before_exposure:
            raise ValueError("late masking cannot establish a clean causal counterfactual")
        if active_correct and not ablated_correct:
            record.status, record.incorporation = MailStatus.INCORPORATED, epoch
            self._release(record)

    ACTIVE = frozenset({MailStatus.PENDING, MailStatus.VISIBLE, MailStatus.ATTENDED})

    def cancel(self, message_id: int) -> None:
        record = self.records[message_id]
        if record.status in self.ACTIVE:
            record.status = MailStatus.CANCELLED
            self._release(record)

    def reject(self, message_id: int) -> None:
        record = self.records[message_id]
        if record.status in self.ACTIVE:
            record.status = MailStatus.REJECTED
            self._release(record)

    def expire(self, epoch: int) -> None:
        for record in self.records.values():
            if epoch >= record.expires and record.status in self.ACTIVE:
                record.status = MailStatus.EXPIRED
                self._release(record)

    def metrics(self) -> dict:
        counts = {status.value: 0 for status in MailStatus}
        for record in self.records.values():
            counts[record.status.value] += 1
        attended = [r for r in self.records.values() if r.first_attention is not None]
        incorporated = [r for r in self.records.values() if r.incorporation is not None]
        expired = [r for r in self.records.values() if r.status == MailStatus.EXPIRED]
        return {
            "counts": counts, "created": len(self.records),
            "attended_not_incorporated": sum(1 for r in attended if r.incorporation is None),
            "attention_latency_epochs": [r.first_attention - r.created for r in attended],
            "incorporation_latency_epochs": [r.incorporation - r.created for r in incorporated],
            # Expired messages are censored at their TTL, not dropped from the denominator.
            "censored_at_ttl": [r.expires - r.created for r in expired],
        }


def merge_entries(*views: PoolView | None) -> ForeignEntries | None:
    """Concatenate several pinned views (native pool, mail pool) into one ForeignEntries.

    Later views are offset so positions stay strictly increasing; recency then treats
    mail as newer than native memory, which is the intended reading order.
    """
    live = [v for v in views if v is not None and v.positions.numel() > 0]
    if not live:
        return None
    positions, layers, offset = [], {}, 0
    for view in live:
        positions.append(view.positions + offset)
        offset = int(positions[-1][-1]) + 1
        for layer, entry in view.layers.items():
            if layer not in layers:
                layers[layer] = entry
            elif isinstance(entry, KV):
                layers[layer] = KV(torch.cat((layers[layer].k, entry.k)), torch.cat((layers[layer].v, entry.v)))
            else:
                layers[layer] = torch.cat((layers[layer], entry))
    return ForeignEntries(torch.cat(positions), layers)


def mail_publication(session, writer: int, translator: Translator, epoch: int, sequence: int, start: int,
                     canonical: Mapping[int, KV | torch.Tensor]) -> Publication:
    rows = {level: r.detach() for level, r in translator.write(canonical).items()}
    return Publication(session, writer, translator.pool.fingerprint, epoch, sequence, start, rows)
