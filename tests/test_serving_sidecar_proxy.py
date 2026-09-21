"""Sidecar (tap -> pool -> inject across different cache layouts) and the OpenAI-compatible proxy."""
from __future__ import annotations
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import UUID
import pytest
import torch
from drift.core.types import KV
from drift.serving.exchange import load_entries, save_entries
from drift.serving.proxy import serve
from drift.serving.sidecar import SidecarCursor, merge_tp_taps, translate, trim_to_blocks
from drift.serving.store import SessionStore
from drift.translate.pool import Layout, PoolFormat, Translator
from drift.transport.wire2 import FrameCodec2

SESSION = UUID(int=12)
FMT = PoolFormat("pool.v1", 2, 6, "ab" * 32)


def test_trim_keeps_sinks_and_most_recent_in_block_multiples():
    assert trim_to_blocks(torch.arange(20), sinks=2, recent=8, block_size=4).tolist() == [0, 1, 14, 15, 16, 17, 18, 19]
    assert trim_to_blocks(torch.arange(11), sinks=2, recent=8, block_size=4).tolist() == [0, 1, 5, 6, 7, 8, 9, 10]
    assert trim_to_blocks(torch.arange(7), sinks=2, recent=8, block_size=4).tolist() == [0, 1, 5, 6]      # drops oldest recent
    assert trim_to_blocks(torch.arange(3), sinks=2, recent=8, block_size=4).numel() == 0                   # never pads


def test_sidecar_translates_split_kv_taps_into_mla_injections(tmp_path):
    torch.manual_seed(0)
    writer = Translator("qwen", Layout("kv_split", 2, 4), {3: 0, 7: 1}, FMT)
    reader = Translator("glm", Layout("mla_latent", 1, 10), {1: 0, 5: 1}, FMT)
    store = SessionStore(tmp_path / "store", FrameCodec2(b"s" * 32))
    # two tensor-parallel ranks each tapped one KV head
    full = {3: KV(torch.randn(9, 2, 4), torch.randn(9, 2, 4)), 7: KV(torch.randn(9, 2, 4), torch.randn(9, 2, 4))}
    for rank in range(2):
        shard = {k: KV(v.k[:, rank:rank + 1], v.v[:, rank:rank + 1]) for k, v in full.items()}
        save_entries(tmp_path / "x" / str(SESSION) / "tap", f"ctx.rank{rank}", shard, torch.arange(9), {"tp_rank": rank})
    merged, _ = merge_tp_taps(tmp_path / "x" / str(SESSION) / "tap", "ctx", 2)
    assert torch.equal(merged[3].k, full[3].k)
    cursor = SidecarCursor()
    report = translate(tmp_path / "x", SESSION, "ctx", "mem", writer, reader, writer_id=1, epoch=0, cursor=cursor,
                       store=store, sinks=2, recent=6, block_size=4, tp_world=2)
    assert report == {"published": report["published"], "injected": 8, "pool_rows": 9} and cursor.sequence == 1 and cursor.start == 9
    entries, positions, manifest = load_entries(tmp_path / "x" / str(SESSION) / "inject", "mem")
    assert set(entries) == {1, 5} and entries[1].shape == (8, 10) and positions.tolist() == list(range(8))
    with torch.no_grad():
        keep = trim_to_blocks(torch.arange(9), 2, 6, 4)
        expected = reader.read({lvl: rows[keep] for lvl, rows in writer.write(full).items()})
    torch.testing.assert_close(entries[5], expected[5])
    # the audited channel holds exactly one authenticated frame of pool rows
    pubs = store.read_from(SESSION, 1, 0)
    assert len(pubs) == 1 and pubs[0].tokens == 9 and set(pubs[0].levels) == {0, 1}
    other = Translator("glm", Layout("mla_latent", 1, 10), {1: 0, 5: 1}, PoolFormat("pool.v1", 2, 6, "00" * 32))
    with pytest.raises(ValueError, match="different pool formats"):
        translate(tmp_path / "x", SESSION, "ctx", "mem2", writer, other, 1, 1, cursor, store, 2, 6, 4, tp_world=2)


@pytest.fixture
def upstream():
    seen = []

    class Fake(BaseHTTPRequestHandler):
        def log_message(self, *a): return
        def do_GET(self):
            payload = b'{"data":[{"id":"glm"}]}'
            self.send_response(200); self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload)
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body, self.headers.get("Authorization")))
            if self.path == "/tokenize":
                payload = {"tokens": [1, 2, 3], "count": 3}
            elif self.path == "/v1/completions":
                payload = {"id": "c1", "created": 1, "model": body["model"], "choices": [{"index": 0, "text": "hello", "finish_reason": "stop"}], "usage": {"prompt_tokens": len(body["prompt"])}}
            else:
                payload = {"echo": body}
            raw = json.dumps(payload).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", seen
    server.shutdown(); server.server_close()


def call(url, body, headers=None):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_proxy_rewrites_drift_chat_and_passes_everything_else_through(tmp_path, upstream):
    url, seen = upstream
    save_entries(tmp_path / "s1" / "inject", "mem", {1: torch.randn(4, 10)}, torch.arange(4), {})
    proxy = serve("127.0.0.1", 0, url, tmp_path, placeholder_id=99)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{proxy.server_address[1]}"
    try:
        status, plain = call(base + "/v1/chat/completions", {"model": "glm", "messages": [{"role": "user", "content": "hi"}]})
        assert status == 200 and seen[-1][0] == "/v1/chat/completions" and "drift" not in seen[-1][1]
        status, out = call(base + "/v1/chat/completions", {"model": "glm", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 7,
                                                            "drift": {"session": "s1", "inject": "mem", "tap": "t0"}},
                           {"Authorization": "Bearer <REDACTED>"})
        assert status == 200 and out["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
        assert out["drift"] == {"session": "s1", "placeholders": 4, "tap": "t0"}
        path, body, auth = seen[-1]
        assert path == "/v1/completions" and body["prompt"] == [99, 99, 99, 99, 1, 2, 3] and body["max_tokens"] == 7
        assert body["kv_transfer_params"] == {"drift_session": "s1", "drift_inject": "mem", "drift_tap": "t0"}
        assert body["cache_salt"] == "drift:s1:mem" and auth == "Bearer <REDACTED>"
        assert seen[-2][0] == "/tokenize" and seen[-2][1]["add_generation_prompt"] is True
        # hard-off: no inject name -> no placeholders, a stock prompt with only the tap tag
        call(base + "/v1/chat/completions", {"model": "glm", "messages": [{"role": "user", "content": "hi"}], "drift": {"session": "s1", "tap": "t1"}})
        assert seen[-1][1]["prompt"] == [1, 2, 3]
        status, err = call(base + "/v1/chat/completions", {"model": "glm", "messages": [], "drift": {"session": "s1", "inject": "missing"}})
        assert status == 400 and "drift" in err["error"]
        status, err = call(base + "/v1/chat/completions", {"model": "glm", "messages": [], "stream": True, "drift": {"session": "s1"}})
        assert status == 400
    finally:
        proxy.shutdown(); proxy.server_close()
