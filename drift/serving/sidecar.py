"""Translator sidecar between serving connectors (docs/reference/service/SERVING_INTEGRATION.md).

A tapping server writes canonical entries to `<root>/<session>/tap/<name>`. The sidecar turns
them into pool rows with the sender's writer, records the publication as an authenticated
wire-v2 frame (the auditable channel), reads the pool rows with the receiver's reader, trims
to the server's block multiple, and writes `<root>/<session>/inject/<name>` for the receiving
server's connector. Tensor-parallel taps (`<name>.rankR`) are merged by concatenating heads.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID
import torch
from drift.core.types import KV
from drift.serving.exchange import load_entries, ready, save_entries
from drift.serving.store import SessionStore
from drift.translate.pool import Translator
from drift.transport.wire2 import Publication


@dataclass
class SidecarCursor:
    sequence: int = 0
    start: int = 0


def merge_tp_taps(directory: Path, name: str, world: int) -> tuple[dict, torch.Tensor]:
    if world == 1:
        entries, positions, _ = load_entries(directory, name)
        return entries, positions
    parts = []
    for rank in range(world):
        if not ready(directory, f"{name}.rank{rank}"):
            raise FileNotFoundError(f"tap from rank {rank} is not ready")
        parts.append(load_entries(directory, f"{name}.rank{rank}"))
    entries = {}
    for layer in parts[0][0]:
        first = parts[0][0][layer]
        if isinstance(first, KV):
            entries[layer] = KV(torch.cat([p[0][layer].k for p in parts], dim=1), torch.cat([p[0][layer].v for p in parts], dim=1))
        else:
            entries[layer] = first          # MLA latents are replicated across ranks, not sharded
    return entries, parts[0][1]


def trim_to_blocks(positions: torch.Tensor, sinks: int, recent: int, block_size: int) -> torch.Tensor:
    """Indices to keep: the first `sinks` plus the most recent entries, total a multiple of
    `block_size` (drop the oldest recent entries first; never pad)."""
    n = positions.numel()
    keep = [i for i in range(n) if i < sinks or i >= n - recent]
    usable = (len(keep) // block_size) * block_size
    if usable == 0:
        return torch.empty(0, dtype=torch.long)
    drop = len(keep) - usable
    head = [i for i in keep if i < sinks]
    tail = [i for i in keep if i >= sinks]
    tail = tail[drop:] if drop <= len(tail) else []
    head = head[: usable - len(tail)]
    return torch.tensor(head + tail, dtype=torch.long)


def translate(root: Path, session: UUID, tap_name: str, inject_name: str, writer: Translator, reader: Translator,
              writer_id: int, epoch: int, cursor: SidecarCursor, store: SessionStore,
              sinks: int, recent: int, block_size: int, tp_world: int = 1) -> dict:
    if writer.pool.fingerprint != reader.pool.fingerprint:
        raise ValueError("writer and reader are fitted to different pool formats")
    tap_dir, inject_dir = root / str(session) / "tap", root / str(session) / "inject"
    entries, positions = merge_tp_taps(tap_dir, tap_name, tp_world)
    with torch.no_grad():
        rows = {level: r.detach() for level, r in writer.write(entries).items()}
        pub = Publication(session, writer_id, writer.pool.fingerprint, epoch, cursor.sequence, cursor.start, rows)
        path = store.append(pub)
        keep = trim_to_blocks(positions, sinks, recent, block_size)
        if keep.numel() == 0:
            return {"published": str(path), "injected": 0, "pool_rows": int(positions.numel())}
        foreign = reader.read({level: r[keep] for level, r in rows.items()})
    save_entries(inject_dir, inject_name, foreign, torch.arange(keep.numel()),
                 {"epoch": epoch, "writer": writer_id, "sequence": cursor.sequence, "source_rows": int(positions.numel())})
    cursor.sequence += 1
    cursor.start += int(positions.numel())
    return {"published": str(path), "injected": int(keep.numel()), "pool_rows": int(positions.numel())}
