"""The export reader slices exactly what the connector's writer put into the pages, across both ranks."""
import json
import struct
import numpy as np
import pytest
import torch
from drift.serving.qwen38_export import MAGIC, read_blob, span_rows, span_selector
from drift.serving.qwen38_pages import rotate, write_compressed, write_kv

ATTN = "language_model.model.layers.{}.self_attn.attn"
SELECT = "language_model.model.layers.{}.self_attn.indexer.compressed_key_cache"


def blob(rank, tensors, blocks):
    """Pack pages the way the owner's exporter does: the request's blocks in order, bf16 bits."""
    entries, chunks, offset = [], [], 0
    for name, cache, group in tensors:
        pages = cache[list(blocks[group])].contiguous().view(torch.int16).numpy().view(np.uint16)
        raw = pages.tobytes()
        entries.append({"layer": name, "part": 0, "group": group, "blocks": list(blocks[group]), "first_block_index": 0,
                        "pages_per_block": 1, "registered_shape": list(cache.shape), "shape": list(pages.shape),
                        "dtype": "bfloat16", "offset": offset, "nbytes": len(raw)})
        chunks.append(raw)
        offset += len(raw)
    entries.append({"layer": "language_model.model.layers.0.linear_attn", "skipped": "state"})
    header = json.dumps({"format": "qwen38-handoff-raw-v1", "tp_rank": rank, "tp_size": 2, "tensors": entries, "data_bytes": offset}).encode()
    start = -(-(16 + len(header)) // 4096) * 4096
    return MAGIC + struct.pack("<Q", len(header)) + header + bytes(start - 16 - len(header)) + b"".join(chunks)


def test_rows_written_by_the_connector_read_back_bit_exactly():
    rng = np.random.default_rng(1)
    keys, values = rng.standard_normal((8, 2, 256)).astype(np.float32), rng.standard_normal((8, 2, 256)).astype(np.float32)
    compressed = rng.standard_normal((2, 128)).astype(np.float32)
    rotated = rotate(keys, np.arange(4, 12), 1e7, 64)
    blocks = ((2, 5), (4, 1))
    blobs = []
    for rank in (0, 1):
        attn = {layer: torch.zeros(6, 1, 8, 512, dtype=torch.bfloat16) for layer in (3, 7)}
        select = {layer: torch.zeros(6, 2, 1, 128, dtype=torch.bfloat16) for layer in (3, 7)}
        for layer in (3, 7):
            write_kv(attn[layer], blocks[0], 4, rotated[:, rank:rank + 1], values[:, rank:rank + 1])
            write_compressed(select[layer], blocks[1], 1, compressed)
        tensors = [(ATTN.format(l), attn[l], 0) for l in (3, 7)] + ([(SELECT.format(l), select[l], 1) for l in (3, 7)] if rank == 0 else [])
        blobs.append(read_blob(blob(rank, tensors, blocks)))
    rows = span_rows(blobs[::-1], 4, 8)
    selector = span_selector(blobs[0], 4, 8)
    bf16 = lambda x: torch.from_numpy(x).to(torch.bfloat16).float().numpy()
    for layer in (3, 7):
        np.testing.assert_array_equal(rows[layer][0], bf16(rotated))
        np.testing.assert_array_equal(rows[layer][1], bf16(values))
        np.testing.assert_array_equal(selector[layer], bf16(compressed))


def test_a_missing_rank_or_foreign_blob_is_refused():
    cache = torch.zeros(6, 1, 8, 512, dtype=torch.bfloat16)
    only = read_blob(blob(0, [(ATTN.format(3), cache, 0)], ((0, 1),)))
    with pytest.raises(ValueError, match="one blob per"):
        span_rows([only], 0, 4)
    with pytest.raises(ValueError, match="not a Qwen3.8"):
        read_blob(b"GLM53HND" + bytes(16))
    with pytest.raises(ValueError, match="selector group"):
        span_selector(only, 2, 4)


def test_a_span_beyond_the_exported_pages_is_refused():
    cache = torch.zeros(6, 1, 8, 512, dtype=torch.bfloat16)
    blobs = [read_blob(blob(rank, [(ATTN.format(3), cache, 0)], ((0, 1),))) for rank in (0, 1)]
    with pytest.raises(ValueError, match="do not cover"):
        span_rows(blobs, 12, 8)
