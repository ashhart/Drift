"""Byte rows of per-position caches shaped [pages, slots, row]: find a span's entries, read them out, write them back.

A cache compressed at ratio r holds one entry per r tokens, so span [start, start + tokens) is entries
[start // r, (start + tokens) // r). Rows move as raw bytes in the cache's own format, which keeps an identity round
trip exact without decoding any quantised layout. Files are npz with a JSON manifest naming each tensor's rows.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def span_entries(start: int, tokens: int, ratio: int) -> tuple[int, int]:
    if ratio < 1 or start < 0 or tokens <= 0 or start % ratio or tokens % ratio:
        raise ValueError(f"span {start}..{start + tokens} does not cover whole entries at ratio {ratio}")
    return start // ratio, tokens // ratio


def locate(page_rows, per_page: int, first: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    entries = np.arange(first, first + count)
    if count <= 0 or entries[-1] // per_page >= len(page_rows):
        raise ValueError("the request's pages do not cover the span's entries")
    return np.asarray(page_rows, dtype=np.int64)[entries // per_page], entries % per_page


def _bytes(part):
    import torch
    if part.ndim != 3:
        raise ValueError(f"cache {tuple(part.shape)} is not [pages, slots, row]")
    return part.view(torch.uint8).reshape(part.shape[0], part.shape[1], -1)


def read_rows(part, page_rows, first: int, count: int) -> np.ndarray:
    """uint8 [count, row_bytes]: the entries' bytes exactly as the cache holds them."""
    view = _bytes(part)
    rows, slots = locate(page_rows, int(part.shape[1]), first, count)
    import torch
    index = torch.as_tensor(rows, device=part.device), torch.as_tensor(slots, device=part.device)
    return view[index[0], index[1]].cpu().numpy()


def write_rows(part, page_rows, first: int, data: np.ndarray) -> None:
    """Write uint8 [count, row_bytes] over the entries, in place."""
    import torch
    view = _bytes(part)
    if data.dtype != np.uint8 or data.ndim != 2 or data.shape[1] != view.shape[2]:
        raise ValueError(f"rows {data.shape} {data.dtype} are not uint8 rows of {view.shape[2]} bytes")
    rows, slots = locate(page_rows, int(part.shape[1]), first, data.shape[0])
    view[torch.as_tensor(rows, device=part.device), torch.as_tensor(slots, device=part.device)] = torch.from_numpy(np.ascontiguousarray(data)).to(part.device)


def save(path: Path, rows: dict[str, tuple[int, np.ndarray]]) -> None:
    """rows: tensor name -> (ratio, uint8 [entries, row_bytes])."""
    names = sorted(rows)
    manifest = json.dumps([{"name": name, "ratio": rows[name][0]} for name in names]).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, manifest=np.frombuffer(manifest, dtype=np.uint8), **{f"r{i}": rows[name][1] for i, name in enumerate(names)})
    tmp.replace(path)


def load(path: Path) -> dict[str, tuple[int, np.ndarray]]:
    with np.load(path) as data:
        manifest = json.loads(bytes(data["manifest"]))
        return {entry["name"]: (int(entry["ratio"]), np.asarray(data[f"r{i}"])) for i, entry in enumerate(manifest)}
