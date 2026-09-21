"""Receiver-local, multi-writer pool bank (HIVE_MIND H2, H5, H7, H8, H10).

One bank per receiver. It accepts pool-format publications from any enrolled writer,
validates them against the pinned session contract, reads them ONCE through the
receiver's own Translator into the receiver's canonical layout, and appends. Entries
are keyed by (writer, sequence); no writer can mutate another's entries. The
receiver's own writer id is excluded (it has those entries natively). The retained
window is bounded: the first `sinks` pool slots plus the `recent` most recent slots.

Commit is atomic across levels and validated before any state changes. Readers pin
one PoolView for a whole model step; a later commit never mutates a pinned view.
Single-threaded reference; serve with a lock or RCU handoff.
"""
from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
from uuid import UUID
import torch
from drift.core.types import KV
from drift.transport.wire2 import Publication
from drift.translate.pool import Translator


@dataclass(frozen=True)
class PoolView:
    epoch: int
    positions: torch.Tensor                 # pool slot per retained entry, strictly increasing
    writers: torch.Tensor                   # writer id per retained entry
    layers: Mapping[int, KV | torch.Tensor] # receiver layer -> canonical entries


@dataclass
class _WriterCursor:
    next_sequence: int = 0
    next_start: int = 0
    last_epoch: int = -1


class PoolBank:
    def __init__(self, session: UUID, reader: Translator, self_writer: int | None,
                 writers: set[int], sinks: int = 4, recent: int = 128, multiple_per_epoch: bool = False):
        """`multiple_per_epoch` is for mail streams, whose policy allows several messages per
        epoch from one sender; sequence and start still make every publication unique and ordered."""
        if not writers or sinks < 0 or recent <= 0:
            raise ValueError("invalid bank configuration")
        self.multiple_per_epoch = multiple_per_epoch
        if self_writer is not None and self_writer in writers:
            raise ValueError("a receiver does not read its own writer through the pool")
        self.session, self.reader, self.self_writer = session, reader, self_writer
        self.writers = frozenset(writers)
        self.sinks, self.recent = sinks, recent
        self.cursors = {w: _WriterCursor() for w in writers}
        self.next_slot = 0
        self.last_epoch = -1
        self._view: PoolView | None = None

    @torch.no_grad()
    def commit(self, pub: Publication) -> None:
        pub.check()
        if pub.session != self.session:
            raise ValueError("wrong run/session")
        if pub.fingerprint != self.reader.pool.fingerprint:
            raise ValueError("publication is in a different pool format")
        if pub.writer == self.self_writer:
            raise ValueError("self publication must not enter the receiver's pool bank")
        if pub.writer not in self.writers:
            raise ValueError("unknown writer; enroll it in the session contract first")
        cursor = self.cursors[pub.writer]
        if pub.sequence != cursor.next_sequence or pub.start != cursor.next_start:
            raise ValueError("duplicate, gap, reorder, or missing full-reset handshake")
        too_old = pub.epoch < cursor.last_epoch if self.multiple_per_epoch else pub.epoch <= cursor.last_epoch
        if too_old or pub.epoch < self.last_epoch:
            raise ValueError("nonmonotonic publication epoch")
        expected = set(map(int, self.reader.readers))
        if set(pub.levels) != expected:
            raise ValueError("publication is missing or adding a pool level")
        converted = self.reader.read({level: rows.to(next(self.reader.parameters())) for level, rows in pub.levels.items()})
        for entry in converted.values():
            if isinstance(entry, KV):
                entry.check()
            elif not torch.isfinite(entry).all():
                raise ValueError("nonfinite projected entry")
        device = next(iter(pub.levels.values())).device
        n = pub.tokens
        positions = torch.arange(self.next_slot, self.next_slot + n, device=device)
        writers = torch.full((n,), pub.writer, dtype=torch.long, device=device)
        if self._view is not None:
            positions = torch.cat((self._view.positions, positions))
            writers = torch.cat((self._view.writers, writers))
            converted = {layer: _cat(self._view.layers[layer], entry) for layer, entry in converted.items()}
        keep = (positions < self.sinks) | (positions >= positions[-1] - self.recent + 1)
        trimmed = {layer: _select(entry, keep) for layer, entry in converted.items()}
        new_view = PoolView(pub.epoch, positions[keep], writers[keep], MappingProxyType(trimmed))
        # All validation, projection and allocation happened before these updates.
        self._view = new_view
        cursor.next_sequence += 1
        cursor.next_start += n
        cursor.last_epoch = pub.epoch
        self.last_epoch = max(self.last_epoch, pub.epoch)
        self.next_slot += n

    def pin(self) -> PoolView | None:
        return self._view

    def snapshot(self) -> dict:
        view = self._view
        return {
            "cursors": {w: (c.next_sequence, c.next_start, c.last_epoch) for w, c in self.cursors.items()},
            "next_slot": self.next_slot, "last_epoch": self.last_epoch,
            "view": None if view is None else {
                "epoch": view.epoch, "positions": view.positions.clone(), "writers": view.writers.clone(),
                "layers": {layer: (entry.clone() if isinstance(entry, KV) else entry.clone())
                           for layer, entry in view.layers.items()}},
        }

    def restore(self, snapshot: dict) -> None:
        if set(snapshot["cursors"]) != set(self.cursors):
            raise ValueError("snapshot writer set differs from the session contract")
        for w, (seq, start, epoch) in snapshot["cursors"].items():
            self.cursors[w] = _WriterCursor(seq, start, epoch)
        self.next_slot, self.last_epoch = snapshot["next_slot"], snapshot["last_epoch"]
        view = snapshot["view"]
        self._view = None if view is None else PoolView(
            view["epoch"], view["positions"].clone(), view["writers"].clone(),
            MappingProxyType({layer: (entry.clone() if isinstance(entry, KV) else entry.clone())
                              for layer, entry in view["layers"].items()}))


def _cat(old: KV | torch.Tensor, new: KV | torch.Tensor) -> KV | torch.Tensor:
    if isinstance(old, KV):
        return KV(torch.cat((old.k, new.k)), torch.cat((old.v, new.v)))
    return torch.cat((old, new))


def _select(entry: KV | torch.Tensor, keep: torch.Tensor) -> KV | torch.Tensor:
    if isinstance(entry, KV):
        return KV(entry.k[keep], entry.v[keep])
    return entry[keep]
