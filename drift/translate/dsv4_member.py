"""DeepSeek V4 in the shared space: its compressed cache entries decoded from hub vectors. NumPy only.

DeepSeek keeps one entry per 4 tokens in its ratio-4 layers, each with an indexer key, and one per 128 tokens in its
ratio-128 layers (docs/evaluation/DSV4_TRANSLATOR_DEVELOPMENT.md). Its decoder works per entry. At ratio 4 the input is
the sender's hub vector under each of the entry's 4 tokens side by side (`slots`), since the entry weighs its tokens by
position; at ratio 128 it is the mean hub vector of the sender's tokens ending inside the entry's characters (`pool`).
Ridge maps take the input to the entry's 448 content values in every layer of that ratio, and at ratio 4 to its
indexer keys too. The 64 rope values carry position, not content, so `inject_rows` keeps DeepSeek's own values for
placeholder text at the same positions. A span starts at a page boundary and the passage fills its first tokens, so
entry e covers the passage's tokens [e * ratio, (e + 1) * ratio). Fitting is in scripts/live/hub_fit_dsv4.py.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from drift.translate.dsv4_pages import FP8_DIMS, encode_entries, encode_index_keys, entries, index_keys

LAYER = re.compile(r"model\.layers\.(\d+)\.attn(\.indexer\.k_cache)?$")
PER_PAGE = {4: 64, 128: 2}                                          # one page holds 256 tokens at either ratio


def names(rows: dict) -> dict:
    """{name: (ratio, rows)} -> {"mla4": {layer: name}, "index4": {layer: name}, "mla128": {layer: name}}."""
    out = {"mla4": {}, "index4": {}, "mla128": {}}
    for name, (ratio, _) in rows.items():
        match = LAYER.match(name)
        if match:
            out["index4" if match.group(2) else f"mla{ratio}"][int(match.group(1))] = name
    return out


def content(rows: dict) -> dict:
    """A span's decoded content: "mla4" [entries, layers, 448], "index4" [entries, layers, 128], "mla128" [entries, layers, 448]."""
    found = names(rows)
    decode = {"mla4": lambda r: entries(r, PER_PAGE[4])[:, :FP8_DIMS], "index4": lambda r: index_keys(r, PER_PAGE[4]),
              "mla128": lambda r: entries(r, PER_PAGE[128])[:, :FP8_DIMS]}
    return {kind: np.stack([decode[kind](rows[found[kind][layer]][1]) for layer in sorted(found[kind])], axis=1) for kind in decode}


def spans(offsets: np.ndarray, count: int, ratio: int, total: int, whole: bool) -> list:
    """Each entry's characters, (start, end), over the passage's tokens; None where it holds none, or with whole=True
    where it holds fewer than `ratio`."""
    out = []
    for e in range(total):
        first, last = e * ratio, min((e + 1) * ratio, count) - 1
        out.append(None if first >= count or (whole and last < (e + 1) * ratio - 1) else (int(offsets[first, 0]), int(offsets[last, 1])))
    return out


def pool(z: np.ndarray, source_offsets: np.ndarray, entry_spans: list) -> tuple[np.ndarray, np.ndarray]:
    """The mean hub vector of the sender's tokens ending inside each entry's characters, and which entries had any."""
    ends = np.asarray(source_offsets)[:, 1]
    pooled, kept = [], []
    for e, span in enumerate(entry_spans):
        if span is None:
            continue
        inside = (ends > span[0]) & (ends <= span[1])
        if inside.any():
            pooled.append(np.asarray(z, np.float32)[inside].mean(0)); kept.append(e)
    width = np.asarray(z).shape[1]
    return (np.stack(pooled) if pooled else np.zeros((0, width), np.float32)), np.asarray(kept, np.int64)


def slots(z: np.ndarray, source_offsets: np.ndarray, offsets: np.ndarray, count: int, ratio: int, total: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Per entry, the first `width` hub values of the sender's token under each of its `ratio` tokens, side by side:
    the sender token that ends at or next after each receiver token's end, zeros past the passage. Entries whose
    tokens are all past the passage are left out."""
    z, ends = np.asarray(z, np.float32)[:, :width], np.asarray(source_offsets)[:, 1]
    rows, kept = [], []
    for e in range(total):
        if e * ratio >= count:
            continue
        row = np.zeros((ratio, width), np.float32)
        for j, token in enumerate(range(e * ratio, min((e + 1) * ratio, count))):
            k = int(np.searchsorted(ends, offsets[token, 1]))
            row[j] = z[min(k, len(z) - 1)]
        rows.append(row.reshape(-1)); kept.append(e)
    return (np.stack(rows) if rows else np.zeros((0, ratio * width), np.float32)), np.asarray(kept, np.int64)


@dataclass
class Decoder:
    """Ridge maps from an entry's input to its content, with each output's spread restored about its centre."""
    layers4: np.ndarray
    layers128: np.ndarray
    w4: np.ndarray
    b4: np.ndarray
    gain4: np.ndarray
    centre4: np.ndarray
    w128: np.ndarray
    b128: np.ndarray
    gain128: np.ndarray
    centre128: np.ndarray

    def ratio4(self, pooled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """[k, inputs] -> MLA content [k, layers, 448] and indexer keys [k, layers, 128]."""
        y = np.asarray(pooled, np.float32) @ self.w4 + self.b4
        y = (self.centre4 + self.gain4 * (y - self.centre4)).reshape(len(y), len(self.layers4), FP8_DIMS + 128)
        return y[:, :, :FP8_DIMS], y[:, :, FP8_DIMS:]

    def ratio128(self, pooled: np.ndarray) -> np.ndarray:
        """[k, inputs] -> MLA content [k, layers, 448]."""
        y = np.asarray(pooled, np.float32) @ self.w128 + self.b128
        return (self.centre128 + self.gain128 * (y - self.centre128)).reshape(len(y), len(self.layers128), FP8_DIMS)


def save(decoder: Decoder, folder: Path, report: dict) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "decoder.npz", **vars(decoder))
    (folder / "decoder.json").write_text(json.dumps(report, indent=1) + "\n")


def load(folder: Path) -> Decoder:
    arrays = np.load(Path(folder) / "decoder.npz")
    return Decoder(**{k: arrays[k] for k in arrays.files})


def inject_rows(blank: dict, mla4: np.ndarray, index4: np.ndarray, kept4: np.ndarray, mla128: np.ndarray, kept128: np.ndarray) -> dict:
    """Raw rows to inject: the placeholder span's own rows with the kept entries' content replaced by translated content.
    Rope values and every other entry stay as DeepSeek wrote them for the placeholders."""
    found, out = names(blank), dict(blank)
    for kind, values, kept in (("mla4", mla4, kept4), ("index4", index4, kept4), ("mla128", mla128, kept128)):
        for column, layer in enumerate(sorted(found[kind])):
            name = found[kind][layer]
            ratio, rows = blank[name]
            if kind == "index4":
                keys = index_keys(rows, PER_PAGE[4])
                keys[kept] = values[:, column]
                out[name] = (ratio, encode_index_keys(keys, PER_PAGE[4]))
            else:
                per_page = PER_PAGE[ratio]
                full = entries(rows, per_page)
                full[kept, :FP8_DIMS] = values[:, column]
                out[name] = (ratio, encode_entries(full, per_page))
    return out
